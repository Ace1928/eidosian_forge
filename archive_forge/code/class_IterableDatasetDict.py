import contextlib
import copy
import fnmatch
import json
import math
import posixpath
import re
import warnings
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import fsspec
import numpy as np
from huggingface_hub import (
from . import config
from .arrow_dataset import PUSH_TO_HUB_WITHOUT_METADATA_CONFIGS_SPLIT_PATTERN_SHARDED, Dataset
from .features import Features
from .features.features import FeatureType
from .info import DatasetInfo, DatasetInfosDict
from .naming import _split_re
from .splits import NamedSplit, Split, SplitDict, SplitInfo
from .table import Table
from .tasks import TaskTemplate
from .utils import logging
from .utils.deprecation_utils import deprecated
from .utils.doc_utils import is_documented_by
from .utils.hub import list_files_info
from .utils.metadata import MetadataConfigs
from .utils.py_utils import asdict, glob_pattern_to_regex, string_to_dict
from .utils.typing import PathLike
class IterableDatasetDict(dict):

    def __repr__(self):
        repr = '\n'.join([f'{k}: {v}' for k, v in self.items()])
        repr = re.sub('^', ' ' * 4, repr, 0, re.M)
        return f'IterableDatasetDict({{\n{repr}\n}})'

    def with_format(self, type: Optional[str]=None) -> 'IterableDatasetDict':
        """
        Return a dataset with the specified format.
        This method only supports the "torch" format for now.
        The format is set to all the datasets of the dataset dictionary.

        Args:
            type (`str`, *optional*, defaults to `None`):
                If set to "torch", the returned dataset
                will be a subclass of `torch.utils.data.IterableDataset` to be used in a `DataLoader`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", streaming=True)
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> def encode(example):
        ...     return tokenizer(examples["text"], truncation=True, padding="max_length")
        >>> ds = ds.map(encode, batched=True, remove_columns=["text"])
        >>> ds = ds.with_format("torch")
        ```
        """
        return IterableDatasetDict({k: dataset.with_format(type=type) for k, dataset in self.items()})

    def map(self, function: Optional[Callable]=None, with_indices: bool=False, input_columns: Optional[Union[str, List[str]]]=None, batched: bool=False, batch_size: int=1000, drop_last_batch: bool=False, remove_columns: Optional[Union[str, List[str]]]=None, fn_kwargs: Optional[dict]=None) -> 'IterableDatasetDict':
        """
        Apply a function to all the examples in the iterable dataset (individually or in batches) and update them.
        If your function returns a column that already exists, then it overwrites it.
        The function is applied on-the-fly on the examples when iterating over the dataset.
        The transformation is applied to all the datasets of the dataset dictionary.

        You can specify whether the function should be batched or not with the `batched` parameter:

        - If batched is `False`, then the function takes 1 example in and should return 1 example.
          An example is a dictionary, e.g. `{"text": "Hello there !"}`.
        - If batched is `True` and `batch_size` is 1, then the function takes a batch of 1 example as input and can return a batch with 1 or more examples.
          A batch is a dictionary, e.g. a batch of 1 example is `{"text": ["Hello there !"]}`.
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
            input_columns (`[Union[str, List[str]]]`, *optional*, defaults to `None`):
                The columns to be passed into `function`
                as positional arguments. If `None`, a dict mapping to all formatted columns is passed as one argument.
            batched (`bool`, defaults to `False`):
                Provide batch of examples to `function`.
            batch_size (`int`, *optional*, defaults to `1000`):
                Number of examples per batch provided to `function` if `batched=True`.
            drop_last_batch (`bool`, defaults to `False`):
                Whether a last batch smaller than the `batch_size` should be
                dropped instead of being processed by the function.
            remove_columns (`[List[str]]`, *optional*, defaults to `None`):
                Remove a selection of columns while doing the mapping.
                Columns will be removed before updating the examples with the output of `function`, i.e. if `function` is adding
                columns with names in `remove_columns`, these columns will be kept.
            fn_kwargs (`Dict`, *optional*, defaults to `None`):
                Keyword arguments to be passed to `function`

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", streaming=True)
        >>> def add_prefix(example):
        ...     example["text"] = "Review: " + example["text"]
        ...     return example
        >>> ds = ds.map(add_prefix)
        >>> next(iter(ds["train"]))
        {'label': 1,
         'text': 'Review: the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'}
        ```
        """
        return IterableDatasetDict({k: dataset.map(function=function, with_indices=with_indices, input_columns=input_columns, batched=batched, batch_size=batch_size, drop_last_batch=drop_last_batch, remove_columns=remove_columns, fn_kwargs=fn_kwargs) for k, dataset in self.items()})

    def filter(self, function: Optional[Callable]=None, with_indices=False, input_columns: Optional[Union[str, List[str]]]=None, batched: bool=False, batch_size: Optional[int]=1000, fn_kwargs: Optional[dict]=None) -> 'IterableDatasetDict':
        """Apply a filter function to all the elements so that the dataset only includes examples according to the filter function.
        The filtering is done on-the-fly when iterating over the dataset.
        The filtering is applied to all the datasets of the dataset dictionary.

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
                Provide batch of examples to `function`
            batch_size (`int`, *optional*, defaults to `1000`):
                Number of examples per batch provided to `function` if `batched=True`.
            fn_kwargs (`Dict`, *optional*, defaults to `None`):
                Keyword arguments to be passed to `function`

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", streaming=True)
        >>> ds = ds.filter(lambda x: x["label"] == 0)
        >>> list(ds["train"].take(3))
        [{'label': 0, 'text': 'Review: simplistic , silly and tedious .'},
         {'label': 0,
         'text': "Review: it's so laddish and juvenile , only teenage boys could possibly find it funny ."},
         {'label': 0,
         'text': 'Review: exploitative and largely devoid of the depth or sophistication that would make watching such a graphic treatment of the crimes bearable .'}]
        ```
        """
        return IterableDatasetDict({k: dataset.filter(function=function, with_indices=with_indices, input_columns=input_columns, batched=batched, batch_size=batch_size, fn_kwargs=fn_kwargs) for k, dataset in self.items()})

    def shuffle(self, seed=None, generator: Optional[np.random.Generator]=None, buffer_size: int=1000) -> 'IterableDatasetDict':
        """
        Randomly shuffles the elements of this dataset.
        The shuffling is applied to all the datasets of the dataset dictionary.

        This dataset fills a buffer with buffer_size elements, then randomly samples elements from this buffer,
        replacing the selected elements with new elements. For perfect shuffling, a buffer size greater than or
        equal to the full size of the dataset is required.

        For instance, if your dataset contains 10,000 elements but `buffer_size` is set to 1000, then `shuffle` will
        initially select a random element from only the first 1000 elements in the buffer. Once an element is
        selected, its space in the buffer is replaced by the next (i.e. 1,001-st) element,
        maintaining the 1000 element buffer.

        If the dataset is made of several shards, it also does `shuffle` the order of the shards.
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
        >>> ds = load_dataset("rotten_tomatoes", streaming=True)
        >>> list(ds["train"].take(3))
        [{'label': 1,
         'text': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'},
         {'label': 1,
         'text': 'the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .'},
         {'label': 1, 'text': 'effective but too-tepid biopic'}]
        >>> ds = ds.shuffle(seed=42)
        >>> list(ds["train"].take(3))
        [{'label': 1,
         'text': "a sports movie with action that's exciting on the field and a story you care about off it ."},
         {'label': 1,
         'text': 'at its best , the good girl is a refreshingly adult take on adultery . . .'},
         {'label': 1,
         'text': "sam jones became a very lucky filmmaker the day wilco got dropped from their record label , proving that one man's ruin may be another's fortune ."}]
        ```
        """
        return IterableDatasetDict({k: dataset.shuffle(seed=seed, generator=generator, buffer_size=buffer_size) for k, dataset in self.items()})

    def rename_column(self, original_column_name: str, new_column_name: str) -> 'IterableDatasetDict':
        """
        Rename a column in the dataset, and move the features associated to the original column under the new column
        name.
        The renaming is applied to all the datasets of the dataset dictionary.

        Args:
            original_column_name (`str`):
                Name of the column to rename.
            new_column_name (`str`):
                New name for the column.

        Returns:
            [`IterableDatasetDict`]: A copy of the dataset with a renamed column.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", streaming=True)
        >>> ds = ds.rename_column("text", "movie_review")
        >>> next(iter(ds["train"]))
        {'label': 1,
         'movie_review': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'}
        ```
        """
        return IterableDatasetDict({k: dataset.rename_column(original_column_name=original_column_name, new_column_name=new_column_name) for k, dataset in self.items()})

    def rename_columns(self, column_mapping: Dict[str, str]) -> 'IterableDatasetDict':
        """
        Rename several columns in the dataset, and move the features associated to the original columns under
        the new column names.
        The renaming is applied to all the datasets of the dataset dictionary.

        Args:
            column_mapping (`Dict[str, str]`):
                A mapping of columns to rename to their new names.

        Returns:
            [`IterableDatasetDict`]: A copy of the dataset with renamed columns

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", streaming=True)
        >>> ds = ds.rename_columns({"text": "movie_review", "label": "rating"})
        >>> next(iter(ds["train"]))
        {'movie_review': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .',
         'rating': 1}
        ```
        """
        return IterableDatasetDict({k: dataset.rename_columns(column_mapping=column_mapping) for k, dataset in self.items()})

    def remove_columns(self, column_names: Union[str, List[str]]) -> 'IterableDatasetDict':
        """
        Remove one or several column(s) in the dataset and the features associated to them.
        The removal is done on-the-fly on the examples when iterating over the dataset.
        The removal is applied to all the datasets of the dataset dictionary.


        Args:
            column_names (`Union[str, List[str]]`):
                Name of the column(s) to remove.

        Returns:
            [`IterableDatasetDict`]: A copy of the dataset object without the columns to remove.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", streaming=True)
        >>> ds = ds.remove_columns("label")
        >>> next(iter(ds["train"]))
        {'text': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'}
        ```
        """
        return IterableDatasetDict({k: dataset.remove_columns(column_names) for k, dataset in self.items()})

    def select_columns(self, column_names: Union[str, List[str]]) -> 'IterableDatasetDict':
        """Select one or several column(s) in the dataset and the features
        associated to them. The selection is done on-the-fly on the examples
        when iterating over the dataset. The selection is applied to all the
        datasets of the dataset dictionary.


        Args:
            column_names (`Union[str, List[str]]`):
                Name of the column(s) to keep.

        Returns:
            [`IterableDatasetDict`]: A copy of the dataset object with only selected columns.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", streaming=True)
        >>> ds = ds.select("text")
        >>> next(iter(ds["train"]))
        {'text': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'}
        ```
        """
        return IterableDatasetDict({k: dataset.select_columns(column_names) for k, dataset in self.items()})

    def cast_column(self, column: str, feature: FeatureType) -> 'IterableDatasetDict':
        """Cast column to feature for decoding.
        The type casting is applied to all the datasets of the dataset dictionary.

        Args:
            column (`str`):
                Column name.
            feature ([`Feature`]):
                Target feature.

        Returns:
            [`IterableDatasetDict`]

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", streaming=True)
        >>> ds["train"].features
        {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None),
         'text': Value(dtype='string', id=None)}
        >>> ds = ds.cast_column('label', ClassLabel(names=['bad', 'good']))
        >>> ds["train"].features
        {'label': ClassLabel(num_classes=2, names=['bad', 'good'], id=None),
         'text': Value(dtype='string', id=None)}
        ```
        """
        return IterableDatasetDict({k: dataset.cast_column(column=column, feature=feature) for k, dataset in self.items()})

    def cast(self, features: Features) -> 'IterableDatasetDict':
        """
        Cast the dataset to a new set of features.
        The type casting is applied to all the datasets of the dataset dictionary.

        Args:
            features (`Features`):
                New features to cast the dataset to.
                The name of the fields in the features must match the current column names.
                The type of the data must also be convertible from one type to the other.
                For non-trivial conversion, e.g. `string` <-> `ClassLabel` you should use [`map`] to update the Dataset.

        Returns:
            [`IterableDatasetDict`]: A copy of the dataset with casted features.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes", streaming=True)
        >>> ds["train"].features
        {'label': ClassLabel(num_classes=2, names=['neg', 'pos'], id=None),
         'text': Value(dtype='string', id=None)}
        >>> new_features = ds["train"].features.copy()
        >>> new_features['label'] = ClassLabel(names=['bad', 'good'])
        >>> new_features['text'] = Value('large_string')
        >>> ds = ds.cast(new_features)
        >>> ds["train"].features
        {'label': ClassLabel(num_classes=2, names=['bad', 'good'], id=None),
         'text': Value(dtype='large_string', id=None)}
        ```
        """
        return IterableDatasetDict({k: dataset.cast(features=features) for k, dataset in self.items()})