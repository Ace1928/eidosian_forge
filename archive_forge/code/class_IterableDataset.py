import copy
import itertools
import sys
import warnings
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import cycle, islice
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import numpy as np
import pyarrow as pa
from . import config
from .arrow_dataset import Dataset, DatasetInfoMixin
from .features import Features
from .features.features import FeatureType, _align_features, _check_if_features_can_be_aligned, cast_to_python_objects
from .filesystems import _reset_fsspec_lock
from .formatting import PythonFormatter, TensorFormatter, get_format_type_from_alias, get_formatter
from .info import DatasetInfo
from .splits import NamedSplit
from .table import cast_table_to_features, read_schema_from_file, table_cast
from .utils.logging import get_logger
from .utils.py_utils import Literal
from .utils.sharding import _merge_gen_kwargs, _number_of_shards_in_gen_kwargs, _shuffle_gen_kwargs, _split_gen_kwargs
class IterableDataset(DatasetInfoMixin):
    """A Dataset backed by an iterable."""

    def __init__(self, ex_iterable: _BaseExamplesIterable, info: Optional[DatasetInfo]=None, split: Optional[NamedSplit]=None, formatting: Optional[FormattingConfig]=None, shuffling: Optional[ShufflingConfig]=None, distributed: Optional[DistributedConfig]=None, token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]]=None, format_type='deprecated'):
        if distributed and distributed.world_size > 1 and shuffling and (shuffling._original_seed is None):
            raise RuntimeError("The dataset doesn't have a fixed random seed across nodes to shuffle and split the list of dataset shards by node. Please pass e.g. `seed=42` in `.shuffle()` to make all the nodes use the same seed. ")
        if format_type != 'deprecated':
            warning_msg = "'format_type' is deprecated and will be removed in the next major version of datasets. "
            help_message = "Please use 'formatting=FormattingConfig(format_type=format_type)' instead."
            warnings.warn(warning_msg + help_message, category=FutureWarning, stacklevel=2)
            formatting = FormattingConfig(format_type=format_type)
        info = info.copy() if info is not None else DatasetInfo()
        DatasetInfoMixin.__init__(self, info=info, split=split)
        self._ex_iterable = ex_iterable
        self._formatting = formatting
        self._shuffling = shuffling
        self._distributed = distributed
        self._epoch = 0
        self._token_per_repo_id: Dict[str, Union[str, bool, None]] = token_per_repo_id or {}
        _maybe_add_torch_iterable_dataset_parent_class(self.__class__)

    def __repr__(self):
        return f'IterableDataset({{\n    features: {(list(self._info.features.keys()) if self._info.features is not None else 'Unknown')},\n    n_shards: {self.n_shards}\n}})'

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        _maybe_add_torch_iterable_dataset_parent_class(self.__class__)

    def _head(self, n=5):
        return _examples_to_batch(list(self.take(n)))

    def _effective_generator(self):
        if self._shuffling and self._epoch == 0:
            return self._shuffling.generator
        elif self._shuffling:
            effective_seed = deepcopy(self._shuffling.generator).integers(0, 1 << 63) - self._epoch
            effective_seed = (1 << 63) + effective_seed if effective_seed < 0 else effective_seed
            return np.random.default_rng(effective_seed)
        else:
            raise ValueError('This dataset is not shuffled')

    @property
    def n_shards(self) -> int:
        if self._distributed and self._ex_iterable.n_shards % self._distributed.world_size == 0:
            return self._ex_iterable.n_shards // self._distributed.world_size
        return self._ex_iterable.n_shards

    def _iter_pytorch(self):
        ex_iterable = self._prepare_ex_iterable_for_iteration()
        _reset_fsspec_lock()
        import torch.utils.data
        worker_info = torch.utils.data.get_worker_info()
        if self._is_main_process() and ex_iterable.n_shards < worker_info.num_workers:
            logger.warning(f'Too many dataloader workers: {worker_info.num_workers} (max is dataset.n_shards={ex_iterable.n_shards}). Stopping {worker_info.num_workers - ex_iterable.n_shards} dataloader workers.')
            logger.info(f"To parallelize data loading, we give each process some shards (or data sources) to process. Therefore it's unnecessary to have a number of workers greater than dataset.n_shards={ex_iterable.n_shards}. To enable more parallelism, please split the dataset in more files than {ex_iterable.n_shards}.")
        _log_prefix = f'node#{self._distributed.rank} ' if self._distributed else ''
        shards_indices = ex_iterable.split_shard_indices_by_worker(worker_info.id, worker_info.num_workers)
        if shards_indices:
            logger.debug(f"{_log_prefix}dataloader worker#{worker_info.id}, ': Starting to iterate over {len(shards_indices)}/{ex_iterable.n_shards} shards.")
            ex_iterable = ex_iterable.shard_data_sources(worker_id=worker_info.id, num_workers=worker_info.num_workers)
            if self._formatting:
                formatter = get_formatter(self._formatting.format_type, features=self.features)
                format_dict = formatter.recursive_tensorize if isinstance(formatter, TensorFormatter) else cast_to_python_objects
            else:
                format_dict = None
            if self._formatting and (ex_iterable.iter_arrow or self._formatting == 'arrow'):
                if ex_iterable.iter_arrow:
                    iterator = _batch_arrow_tables(ex_iterable.iter_arrow(), batch_size=1)
                else:
                    iterator = _convert_to_arrow(ex_iterable, batch_size=1)
                for key, pa_table in iterator:
                    yield formatter.format_row(pa_table)
                return
            else:
                for key, example in ex_iterable:
                    if self.features:
                        example = _apply_feature_types_on_example(example, self.features, token_per_repo_id=self._token_per_repo_id)
                    yield (format_dict(example) if format_dict else example)
            logger.debug(f"{_log_prefix}dataloader worker#{worker_info.id}, ': Finished iterating over {len(shards_indices)}/{ex_iterable.n_shards} shards.")
        else:
            logger.debug(f"{_log_prefix}dataloader worker#{worker_info.id}, ': Stopping... Number of dataset shards < num_workers ({ex_iterable.n_shards}<{worker_info.num_workers}).")

    def _is_main_process(self):
        if self._distributed and self._distributed.rank > 0:
            return False
        if 'torch' in sys.modules:
            import torch.utils.data
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None and worker_info.id > 0:
                return False
        return True

    def _prepare_ex_iterable_for_iteration(self) -> _BaseExamplesIterable:
        if self._shuffling:
            ex_iterable = self._ex_iterable.shuffle_data_sources(self._effective_generator())
        else:
            ex_iterable = self._ex_iterable
        if self._distributed:
            rank = self._distributed.rank
            world_size = self._distributed.world_size
            if ex_iterable.n_shards % world_size == 0:
                if self._is_main_process():
                    n_shards_per_node = ex_iterable.n_shards // world_size
                    plural = 's' if n_shards_per_node > 1 else ''
                    logger.info(f'Assigning {n_shards_per_node} shard{plural} (or data source{plural}) of the dataset to each node.')
                ex_iterable = ex_iterable.shard_data_sources(rank, world_size)
            else:
                if self._is_main_process():
                    logger.info(f'Assigning 1 out of {world_size} examples of the dataset to each node. The others are skipped during the iteration.')
                    logger.info(f'It is more optimized to distribute the dataset shards (or data sources) across nodes. You can do that by using a dataset with number of shards that is a factor of world_size={world_size}. The current dataset has {ex_iterable.n_shards} which is not a factor of {world_size}')
                ex_iterable = StepExamplesIterable(ex_iterable, step=world_size, offset=rank)
        return ex_iterable

    def __iter__(self):
        if 'torch' in sys.modules:
            import torch.utils.data
            worker_info = torch.utils.data.get_worker_info()
            if isinstance(self, torch.utils.data.IterableDataset) and worker_info is not None:
                yield from self._iter_pytorch()
                return
        ex_iterable = self._prepare_ex_iterable_for_iteration()
        if self._formatting:
            formatter = get_formatter(self._formatting.format_type, features=self.features)
            format_dict = formatter.recursive_tensorize if isinstance(formatter, TensorFormatter) else cast_to_python_objects
        else:
            format_dict = None
        if self._formatting and (ex_iterable.iter_arrow or self._formatting.format_type == 'arrow'):
            if ex_iterable.iter_arrow:
                iterator = _batch_arrow_tables(ex_iterable.iter_arrow(), batch_size=1)
            else:
                iterator = _convert_to_arrow(ex_iterable, batch_size=1)
            for key, pa_table in iterator:
                yield formatter.format_row(pa_table)
            return
        for key, example in ex_iterable:
            if self.features:
                example = _apply_feature_types_on_example(example, self.features, token_per_repo_id=self._token_per_repo_id)
            yield (format_dict(example) if format_dict else example)

    def iter(self, batch_size: int, drop_last_batch: bool=False):
        """Iterate through the batches of size `batch_size`.

        Args:
            batch_size (:obj:`int`): size of each batch to yield.
            drop_last_batch (:obj:`bool`, default `False`): Whether a last batch smaller than the batch_size should be
                dropped
        """
        if self._formatting:
            formatter = get_formatter(self._formatting.format_type, features=self.features)
            format_dict = formatter.recursive_tensorize if isinstance(formatter, TensorFormatter) else cast_to_python_objects
        else:
            format_dict = None
        ex_iterable = self._prepare_ex_iterable_for_iteration()
        if self._formatting and (ex_iterable.iter_arrow or self._formatting == 'arrow'):
            if ex_iterable.iter_arrow:
                iterator = _batch_arrow_tables(ex_iterable.iter_arrow(), batch_size=batch_size, drop_last_batch=drop_last_batch)
            else:
                iterator = _convert_to_arrow(ex_iterable, batch_size=batch_size, drop_last_batch=drop_last_batch)
            for key, pa_table in iterator:
                yield formatter.format_batch(pa_table)
            return
        iterator = iter(ex_iterable)
        for key, example in iterator:
            examples = [example] + [example for key, example in islice(iterator, batch_size - 1)]
            if drop_last_batch and len(examples) < batch_size:
                return
            batch = _examples_to_batch(examples)
            if self.features:
                batch = _apply_feature_types_on_batch(batch, self.features, token_per_repo_id=self._token_per_repo_id)
            yield (format_dict(batch) if format_dict else batch)

    @staticmethod
    def from_generator(generator: Callable, features: Optional[Features]=None, gen_kwargs: Optional[dict]=None) -> 'IterableDataset':
        """Create an Iterable Dataset from a generator.

        Args:
            generator (`Callable`):
                A generator function that `yields` examples.
            features (`Features`, *optional*):
                Dataset features.
            gen_kwargs(`dict`, *optional*):
                Keyword arguments to be passed to the `generator` callable.
                You can define a sharded iterable dataset by passing the list of shards in `gen_kwargs`.
                This can be used to improve shuffling and when iterating over the dataset with multiple workers.

        Returns:
            `IterableDataset`

        Example:

        ```py
        >>> def gen():
        ...     yield {"text": "Good", "label": 0}
        ...     yield {"text": "Bad", "label": 1}
        ...
        >>> ds = IterableDataset.from_generator(gen)
        ```

        ```py
        >>> def gen(shards):
        ...     for shard in shards:
        ...         with open(shard) as f:
        ...             for line in f:
        ...                 yield {"line": line}
        ...
        >>> shards = [f"data{i}.txt" for i in range(32)]
        >>> ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": shards})
        >>> ds = ds.shuffle(seed=42, buffer_size=10_000)  # shuffles the shards order + uses a shuffle buffer
        >>> from torch.utils.data import DataLoader
        >>> dataloader = DataLoader(ds.with_format("torch"), num_workers=4)  # give each worker a subset of 32/4=8 shards
        ```
        """
        from .io.generator import GeneratorDatasetInputStream
        return GeneratorDatasetInputStream(generator=generator, features=features, gen_kwargs=gen_kwargs, streaming=True).read()

    @staticmethod
    def from_spark(df: 'pyspark.sql.DataFrame', split: Optional[NamedSplit]=None, features: Optional[Features]=None, **kwargs) -> 'IterableDataset':
        """Create an IterableDataset from Spark DataFrame. The dataset is streamed to the driver in batches.

        Args:
            df (`pyspark.sql.DataFrame`):
                The DataFrame containing the desired data.
            split (`NamedSplit`, *optional*):
                Split name to be assigned to the dataset.
            features (`Features`, *optional*):
                Dataset features.

        Returns:
            [`IterableDataset`]

        Example:

        ```py
        >>> df = spark.createDataFrame(
        >>>     data=[[1, "Elia"], [2, "Teo"], [3, "Fang"]],
        >>>     columns=["id", "name"],
        >>> )
        >>> ds = IterableDataset.from_spark(df)
        ```
        """
        from .io.spark import SparkDatasetReader
        if sys.platform == 'win32':
            raise EnvironmentError('IterableDataset.from_spark is not currently supported on Windows')
        return SparkDatasetReader(df, split=split, features=features, streaming=True, **kwargs).read()

    @staticmethod
    def from_file(filename: str) -> 'IterableDataset':
        """Instantiate a IterableDataset from Arrow table at filename.

        Args:
            filename (`str`):
                File name of the dataset.

        Returns:
            [`IterableDataset`]
        """
        pa_table_schema = read_schema_from_file(filename)
        inferred_features = Features.from_arrow_schema(pa_table_schema)
        ex_iterable = ArrowExamplesIterable(Dataset._generate_tables_from_cache_file, kwargs={'filename': filename})
        return IterableDataset(ex_iterable=ex_iterable, info=DatasetInfo(features=inferred_features))

    def with_format(self, type: Optional[str]=None) -> 'IterableDataset':
        """
        Return a dataset with the specified format.
        Supported formats: "arrow", or None for regular python objects.
        The other formats are currently not implemented.

        Args:

            type (`str`, optional, default None): if set to "torch", the returned dataset
                will be a subclass of torch.utils.data.IterableDataset to be used in a DataLoader
        """
        type = get_format_type_from_alias(type)
        return IterableDataset(ex_iterable=self._ex_iterable, info=self._info.copy(), split=self._split, formatting=FormattingConfig(format_type=type), shuffling=copy.deepcopy(self._shuffling), distributed=copy.deepcopy(self._distributed), token_per_repo_id=self._token_per_repo_id)

    def map(self, function: Optional[Callable]=None, with_indices: bool=False, input_columns: Optional[Union[str, List[str]]]=None, batched: bool=False, batch_size: Optional[int]=1000, drop_last_batch: bool=False, remove_columns: Optional[Union[str, List[str]]]=None, features: Optional[Features]=None, fn_kwargs: Optional[dict]=None) -> 'IterableDataset':
        """
        Apply a function to all the examples in the iterable dataset (individually or in batches) and update them.
        If your function returns a column that already exists, then it overwrites it.
        The function is applied on-the-fly on the examples when iterating over the dataset.

        You can specify whether the function should be batched or not with the `batched` parameter:

        - If batched is `False`, then the function takes 1 example in and should return 1 example.
          An example is a dictionary, e.g. `{"text": "Hello there !"}`.
        - If batched is `True` and `batch_size` is 1, then the function takes a batch of 1 example as input and can return a batch with 1 or more examples.
          A batch is a dictionary, e.g. a batch of 1 example is {"text": ["Hello there !"]}.
        - If batched is `True` and `batch_size` is `n` > 1, then the function takes a batch of `n` examples as input and can return a batch with `n` examples, or with an arbitrary number of examples.
          Note that the last batch may have less than `n` examples.
          A batch is a dictionary, e.g. a batch of `n` examples is `{"text": ["Hello there !"] * n}`.

        Args:
            function (`Callable`, *optional*, defaults to `None`):
                Function applied on-the-fly on the examples when you iterate on the dataset.
                It must have one of the following signatures:

                - `function(example: Dict[str, Any]) -> Dict[str, Any]` if `batched=False` and `with_indices=False`
                - `function(example: Dict[str, Any], idx: int) -> Dict[str, Any]` if `batched=False` and `with_indices=True`
                - `function(batch: Dict[str, List]) -> Dict[str, List]` if `batched=True` and `with_indices=False`
                - `function(batch: Dict[str, List], indices: List[int]) -> Dict[str, List]` if `batched=True` and `with_indices=True`

                For advanced usage, the function can also return a `pyarrow.Table`.
                Moreover if your function returns nothing (`None`), then `map` will run your function and return the dataset unchanged.
                If no function is provided, default to identity function: `lambda x: x`.
            with_indices (`bool`, defaults to `False`):
                Provide example indices to `function`. Note that in this case the signature of `function` should be `def function(example, idx[, rank]): ...`.
            input_columns (`Optional[Union[str, List[str]]]`, defaults to `None`):
                The columns to be passed into `function`
                as positional arguments. If `None`, a dict mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`):
                Provide batch of examples to `function`.
            batch_size (`int`, *optional*, defaults to `1000`):
                Number of examples per batch provided to `function` if `batched=True`.
                `batch_size <= 0` or `batch_size == None` then provide the full dataset as a single batch to `function`.
            drop_last_batch (`bool`, defaults to `False`):
                Whether a last batch smaller than the batch_size should be
                dropped instead of being processed by the function.
            remove_columns (`[List[str]]`, *optional*, defaults to `None`):
                Remove a selection of columns while doing the mapping.
                Columns will be removed before updating the examples with the output of `function`, i.e. if `function` is adding
                columns with names in `remove_columns`, these columns will be kept.
            features (`[Features]`, *optional*, defaults to `None`):
                Feature types of the resulting dataset.
            fn_kwargs (`Dict`, *optional*, default `None`):
                Keyword arguments to be passed to `function`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train", streaming=True)
        >>> def add_prefix(example):
        ...     example["text"] = "Review: " + example["text"]
        ...     return example
        >>> ds = ds.map(add_prefix)
        >>> list(ds.take(3))
        [{'label': 1,
         'text': 'Review: the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'},
         {'label': 1,
         'text': 'Review: the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .'},
         {'label': 1, 'text': 'Review: effective but too-tepid biopic'}]
        ```
        """
        if isinstance(input_columns, str):
            input_columns = [input_columns]
        if isinstance(remove_columns, str):
            remove_columns = [remove_columns]
        if function is None:
            function = identity_func
        if fn_kwargs is None:
            fn_kwargs = {}
        ex_iterable = MappedExamplesIterable(TypedExamplesIterable(self._ex_iterable, self._info.features, token_per_repo_id=self._token_per_repo_id) if self._info.features is not None else self._ex_iterable, function=function, with_indices=with_indices, input_columns=input_columns, batched=batched, batch_size=batch_size, drop_last_batch=drop_last_batch, remove_columns=remove_columns, fn_kwargs=fn_kwargs, formatting=self._formatting)
        info = self.info.copy()
        info.features = features
        return IterableDataset(ex_iterable=ex_iterable, info=info, split=self._split, formatting=self._formatting, shuffling=copy.deepcopy(self._shuffling), distributed=copy.deepcopy(self._distributed), token_per_repo_id=self._token_per_repo_id)

    def filter(self, function: Optional[Callable]=None, with_indices=False, input_columns: Optional[Union[str, List[str]]]=None, batched: bool=False, batch_size: Optional[int]=1000, fn_kwargs: Optional[dict]=None) -> 'IterableDataset':
        """Apply a filter function to all the elements so that the dataset only includes examples according to the filter function.
        The filtering is done on-the-fly when iterating over the dataset.

        Args:
            function (`Callable`):
                Callable with one of the following signatures:

                - `function(example: Dict[str, Any]) -> bool` if `with_indices=False, batched=False`
                - `function(example: Dict[str, Any], indices: int) -> bool` if `with_indices=True, batched=False`
                - `function(example: Dict[str, List]) -> List[bool]` if `with_indices=False, batched=True`
                - `function(example: Dict[str, List], indices: List[int]) -> List[bool]` if `with_indices=True, batched=True`

                If no function is provided, defaults to an always True function: `lambda x: True`.
            with_indices (`bool`, defaults to `False`):
                Provide example indices to `function`. Note that in this case the signature of `function` should be `def function(example, idx): ...`.
            input_columns (`str` or `List[str]`, *optional*):
                The columns to be passed into `function` as
                positional arguments. If `None`, a dict mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`):
                Provide batch of examples to `function`.
            batch_size (`int`, *optional*, default `1000`):
                Number of examples per batch provided to `function` if `batched=True`.
            fn_kwargs (`Dict`, *optional*, default `None`):
                Keyword arguments to be passed to `function`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train", streaming=True)
        >>> ds = ds.filter(lambda x: x["label"] == 0)
        >>> list(ds.take(3))
        [{'label': 0, 'movie_review': 'simplistic , silly and tedious .'},
         {'label': 0,
         'movie_review': "it's so laddish and juvenile , only teenage boys could possibly find it funny ."},
         {'label': 0,
         'movie_review': 'exploitative and largely devoid of the depth or sophistication that would make watching such a graphic treatment of the crimes bearable .'}]
        ```
        """
        if isinstance(input_columns, str):
            input_columns = [input_columns]
        info = copy.deepcopy(self._info)
        info.features = None
        ex_iterable = FilteredExamplesIterable(TypedExamplesIterable(self._ex_iterable, self._info.features, token_per_repo_id=self._token_per_repo_id) if self._info.features is not None else self._ex_iterable, function=function, with_indices=with_indices, input_columns=input_columns, batched=batched, batch_size=batch_size, fn_kwargs=fn_kwargs, formatting=self._formatting)
        return IterableDataset(ex_iterable=ex_iterable, info=info, split=self._split, formatting=self._formatting, shuffling=copy.deepcopy(self._shuffling), distributed=copy.deepcopy(self._distributed), token_per_repo_id=self._token_per_repo_id)

    def shuffle(self, seed=None, generator: Optional[np.random.Generator]=None, buffer_size: int=1000) -> 'IterableDataset':
        """
        Randomly shuffles the elements of this dataset.

        This dataset fills a buffer with `buffer_size` elements, then randomly samples elements from this buffer,
        replacing the selected elements with new elements. For perfect shuffling, a buffer size greater than or
        equal to the full size of the dataset is required.

        For instance, if your dataset contains 10,000 elements but `buffer_size` is set to 1000, then `shuffle` will
        initially select a random element from only the first 1000 elements in the buffer. Once an element is
        selected, its space in the buffer is replaced by the next (i.e. 1,001-st) element,
        maintaining the 1000 element buffer.

        If the dataset is made of several shards, it also does shuffle the order of the shards.
        However if the order has been fixed by using [`~datasets.IterableDataset.skip`] or [`~datasets.IterableDataset.take`]
        then the order of the shards is kept unchanged.

        Args:
            seed (`int`, *optional*, defaults to `None`):
                Random seed that will be used to shuffle the dataset.
                It is used to sample from the shuffle buffer and also to shuffle the data shards.
            generator (`numpy.random.Generator`, *optional*):
                Numpy random Generator to use to compute the permutation of the dataset rows.
                If `generator=None` (default), uses `np.random.default_rng` (the default BitGenerator (PCG64) of NumPy).
            buffer_size (`int`, defaults to `1000`):
                Size of the buffer.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train", streaming=True)
        >>> list(ds.take(3))
        [{'label': 1,
         'text': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'},
         {'label': 1,
         'text': 'the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .'},
         {'label': 1, 'text': 'effective but too-tepid biopic'}]
        >>> shuffled_ds = ds.shuffle(seed=42)
        >>> list(shuffled_ds.take(3))
        [{'label': 1,
         'text': "a sports movie with action that's exciting on the field and a story you care about off it ."},
         {'label': 1,
         'text': 'at its best , the good girl is a refreshingly adult take on adultery . . .'},
         {'label': 1,
         'text': "sam jones became a very lucky filmmaker the day wilco got dropped from their record label , proving that one man's ruin may be another's fortune ."}]
        ```
        """
        if generator is None:
            generator = np.random.default_rng(seed)
        else:
            generator = deepcopy(generator)
        shuffling = ShufflingConfig(generator=generator, _original_seed=seed)
        return IterableDataset(ex_iterable=BufferShuffledExamplesIterable(self._ex_iterable, buffer_size=buffer_size, generator=generator).shuffle_data_sources(generator), info=self._info.copy(), split=self._split, formatting=self._formatting, shuffling=shuffling, distributed=copy.deepcopy(self._distributed), token_per_repo_id=self._token_per_repo_id)

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def skip(self, n) -> 'IterableDataset':
        """
        Create a new [`IterableDataset`] that skips the first `n` elements.

        Args:
            n (`int`):
                Number of elements to skip.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train", streaming=True)
        >>> list(ds.take(3))
        [{'label': 1,
         'text': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'},
         {'label': 1,
         'text': 'the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .'},
         {'label': 1, 'text': 'effective but too-tepid biopic'}]
        >>> ds = ds.skip(1)
        >>> list(ds.take(3))
        [{'label': 1,
         'text': 'the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .'},
         {'label': 1, 'text': 'effective but too-tepid biopic'},
         {'label': 1,
         'text': 'if you sometimes like to go to the movies to have fun , wasabi is a good place to start .'}]
        ```
        """
        ex_iterable = SkipExamplesIterable(self._ex_iterable, n)
        return IterableDataset(ex_iterable=ex_iterable, info=self._info.copy(), split=self._split, formatting=self._formatting, shuffling=copy.deepcopy(self._shuffling), distributed=copy.deepcopy(self._distributed), token_per_repo_id=self._token_per_repo_id)

    def take(self, n) -> 'IterableDataset':
        """
        Create a new [`IterableDataset`] with only the first `n` elements.

        Args:
            n (`int`):
                Number of elements to take.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train", streaming=True)
        >>> small_ds = ds.take(2)
        >>> list(small_ds)
        [{'label': 1,
         'text': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'},
         {'label': 1,
         'text': 'the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .'}]
        ```
        """
        ex_iterable = TakeExamplesIterable(self._ex_iterable, n)
        return IterableDataset(ex_iterable=ex_iterable, info=self._info.copy(), split=self._split, formatting=self._formatting, shuffling=copy.deepcopy(self._shuffling), distributed=copy.deepcopy(self._distributed), token_per_repo_id=self._token_per_repo_id)

    @property
    def column_names(self) -> Optional[List[str]]:
        """Names of the columns in the dataset.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="validation", streaming=True)
        >>> ds.column_names
        ['text', 'label']
        ```
        """
        return list(self._info.features.keys()) if self._info.features is not None else None

    def add_column(self, name: str, column: Union[list, np.array]) -> 'IterableDataset':
        """Add column to Dataset.

        Args:
            name (str): Column name.
            column (list or np.array): Column data to be added.

        Returns:
            `IterableDataset`
        """
        return self.map(partial(add_column_fn, name=name, column=column), with_indices=True)

    def rename_column(self, original_column_name: str, new_column_name: str) -> 'IterableDataset':
        """
        Rename a column in the dataset, and move the features associated to the original column under the new column
        name.

        Args:
            original_column_name (`str`):
                Name of the column to rename.
            new_column_name (`str`):
                New name for the column.

        Returns:
            `IterableDataset`: A copy of the dataset with a renamed column.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train", streaming=True)
        >>> next(iter(ds))
        {'label': 1,
         'text': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'}
        >>> ds = ds.rename_column("text", "movie_review")
        >>> next(iter(ds))
        {'label': 1,
         'movie_review': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'}
        ```
        """
        return self.rename_columns({original_column_name: new_column_name})

    def rename_columns(self, column_mapping: Dict[str, str]) -> 'IterableDataset':
        """
        Rename several columns in the dataset, and move the features associated to the original columns under
        the new column names.

        Args:
            column_mapping (`Dict[str, str]`): A mapping of columns to rename to their new names

        Returns:
            `IterableDataset`: A copy of the dataset with renamed columns
        """
        original_features = self._info.features.copy() if self._info.features else None
        ds_iterable = self.map(partial(_rename_columns_fn, column_mapping=column_mapping), remove_columns=list(column_mapping))
        if original_features is not None:
            ds_iterable._info.features = Features({column_mapping[col] if col in column_mapping.keys() else col: feature for col, feature in original_features.items()})
            try:
                ds_iterable._info.copy()
            except ValueError:
                ds_iterable._info.task_templates = None
        return ds_iterable

    def remove_columns(self, column_names: Union[str, List[str]]) -> 'IterableDataset':
        """
        Remove one or several column(s) in the dataset and the features associated to them.
        The removal is done on-the-fly on the examples when iterating over the dataset.


        Args:
            column_names (`Union[str, List[str]]`):
                Name of the column(s) to remove.

        Returns:
            `IterableDataset`: A copy of the dataset object without the columns to remove.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train", streaming=True)
        >>> next(iter(ds))
        {'text': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .', 'label': 1}
        >>> ds = ds.remove_columns("label")
        >>> next(iter(ds))
        {'text': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'}
        ```
        """
        original_features = self._info.features.copy() if self._info.features else None
        ds_iterable = self.map(remove_columns=column_names)
        if original_features is not None:
            ds_iterable._info.features = original_features.copy()
            for col, _ in original_features.items():
                if col in column_names:
                    del ds_iterable._info.features[col]
            try:
                ds_iterable._info.copy()
            except ValueError:
                ds_iterable._info.task_templates = None
        return ds_iterable

    def select_columns(self, column_names: Union[str, List[str]]) -> 'IterableDataset':
        """Select one or several column(s) in the dataset and the features
        associated to them. The selection is done on-the-fly on the examples
        when iterating over the dataset.


        Args:
            column_names (`Union[str, List[str]]`):
                Name of the column(s) to select.

        Returns:
            `IterableDataset`: A copy of the dataset object with selected columns.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train", streaming=True)
        >>> next(iter(ds))
        {'text': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .', 'label': 1}
        >>> ds = ds.select_columns("text")
        >>> next(iter(ds))
        {'text': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'}
        ```
        """
        if isinstance(column_names, str):
            column_names = [column_names]
        if self._info:
            info = copy.deepcopy(self._info)
            if self._info.features is not None:
                missing_columns = set(column_names) - set(self._info.features.keys())
                if missing_columns:
                    raise ValueError(f'Column name {list(missing_columns)} not in the dataset. Columns in the dataset: {list(self._info.features.keys())}.')
                info.features = Features({c: info.features[c] for c in column_names})
                try:
                    info.copy()
                except ValueError:
                    info.task_templates = None
        ex_iterable = SelectColumnsIterable(self._ex_iterable, column_names)
        return IterableDataset(ex_iterable=ex_iterable, info=info, split=self._split, formatting=self._formatting, shuffling=self._shuffling, distributed=self._distributed, token_per_repo_id=self._token_per_repo_id)

    def cast_column(self, column: str, feature: FeatureType) -> 'IterableDataset':
        """Cast column to feature for decoding.

        Args:
            column (`str`):
                Column name.
            feature (`Feature`):
                Target feature.

        Returns:
            `IterableDataset`

        Example:

        ```py
        >>> from datasets import load_dataset, Audio
        >>> ds = load_dataset("PolyAI/minds14", name="en-US", split="train", streaming=True)
        >>> ds.features
        {'audio': Audio(sampling_rate=8000, mono=True, decode=True, id=None),
         'english_transcription': Value(dtype='string', id=None),
         'intent_class': ClassLabel(num_classes=14, names=['abroad', 'address', 'app_error', 'atm_limit', 'balance', 'business_loan',  'card_issues', 'cash_deposit', 'direct_debit', 'freeze', 'high_value_payment', 'joint_account', 'latest_transactions', 'pay_bill'], id=None),
         'lang_id': ClassLabel(num_classes=14, names=['cs-CZ', 'de-DE', 'en-AU', 'en-GB', 'en-US', 'es-ES', 'fr-FR', 'it-IT', 'ko-KR',  'nl-NL', 'pl-PL', 'pt-PT', 'ru-RU', 'zh-CN'], id=None),
         'path': Value(dtype='string', id=None),
         'transcription': Value(dtype='string', id=None)}
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        >>> ds.features
        {'audio': Audio(sampling_rate=16000, mono=True, decode=True, id=None),
         'english_transcription': Value(dtype='string', id=None),
         'intent_class': ClassLabel(num_classes=14, names=['abroad', 'address', 'app_error', 'atm_limit', 'balance', 'business_loan',  'card_issues', 'cash_deposit', 'direct_debit', 'freeze', 'high_value_payment', 'joint_account', 'latest_transactions', 'pay_bill'], id=None),
         'lang_id': ClassLabel(num_classes=14, names=['cs-CZ', 'de-DE', 'en-AU', 'en-GB', 'en-US', 'es-ES', 'fr-FR', 'it-IT', 'ko-KR',  'nl-NL', 'pl-PL', 'pt-PT', 'ru-RU', 'zh-CN'], id=None),
         'path': Value(dtype='string', id=None),
         'transcription': Value(dtype='string', id=None)}
        ```
        """
        info = self._info.copy()
        info.features[column] = feature
        try:
            info.copy()
        except ValueError:
            info.task_templates = None
        return IterableDataset(ex_iterable=self._ex_iterable, info=info, split=self._split, formatting=self._formatting, shuffling=copy.deepcopy(self._shuffling), distributed=copy.deepcopy(self._distributed), token_per_repo_id=self._token_per_repo_id)

    def cast(self, features: Features) -> 'IterableDataset':
        """
        Cast the dataset to a new set of features.

        Args:
            features ([`Features`]):
                New features to cast the dataset to.
                The name of the fields in the features must match the current column names.
                The type of the data must also be convertible from one type to the other.
                For non-trivial conversion, e.g. `string` <-> `ClassLabel` you should use [`~Dataset.map`] to update the Dataset.

        Returns:
            `IterableDataset`: A copy of the dataset with casted features.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", split="train", streaming=True)
        >>> ds.features
        {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None),
         'text': Value(dtype='string', id=None)}
        >>> new_features = ds.features.copy()
        >>> new_features["label"] = ClassLabel(names=["bad", "good"])
        >>> new_features["text"] = Value("large_string")
        >>> ds = ds.cast(new_features)
        >>> ds.features
        {'label': ClassLabel(num_classes=2, names=['bad', 'good'], id=None),
         'text': Value(dtype='large_string', id=None)}
        ```
        """
        info = self._info.copy()
        info.features = features
        try:
            info.copy()
        except ValueError:
            info.task_templates = None
        return IterableDataset(ex_iterable=self._ex_iterable, info=info, split=self._split, formatting=self._formatting, shuffling=copy.deepcopy(self._shuffling), distributed=copy.deepcopy(self._distributed), token_per_repo_id=self._token_per_repo_id)

    def _step(self, step: int, offset: int) -> 'IterableDataset':
        ex_iterable = StepExamplesIterable(self._ex_iterable, step=step, offset=offset)
        return IterableDataset(ex_iterable=ex_iterable, info=self._info.copy(), split=self._split, formatting=self._formatting, shuffling=copy.deepcopy(self._shuffling), distributed=copy.deepcopy(self._distributed), token_per_repo_id=self._token_per_repo_id)

    def _resolve_features(self):
        if self.features is not None:
            return self
        elif isinstance(self._ex_iterable, TypedExamplesIterable):
            features = self._ex_iterable.features
        else:
            features = _infer_features_from_batch(self.with_format(None)._head())
        info = self.info.copy()
        info.features = features
        return IterableDataset(ex_iterable=self._ex_iterable, info=info, split=self._split, formatting=self._formatting, shuffling=copy.deepcopy(self._shuffling), distributed=copy.deepcopy(self._distributed), token_per_repo_id=self._token_per_repo_id)