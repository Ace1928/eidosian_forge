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
def _interleave_iterable_datasets(datasets: List[IterableDataset], probabilities: Optional[List[float]]=None, seed: Optional[int]=None, info: Optional[DatasetInfo]=None, split: Optional[NamedSplit]=None, stopping_strategy: Literal['first_exhausted', 'all_exhausted']='first_exhausted') -> IterableDataset:
    """
    Interleave several iterable datasets (sources) into a single iterable dataset.
    The new iterable dataset alternates between the sources to yield examples.
    If `probabilities = None` (default) the iterable dataset will cycles through the sources in order for each next example in the iteration.
    If `probabilities` is not `None, the iterable dataset will sample a random source according to the provided probabilities for each next examples in the iteration.

    <Added version="2.4.0"/>

    Args:
        datasets (`List[IterableDataset]`): list of datasets to interleave
        probabilities (`List[float]`, optional, default None): If specified, the new iterable dataset samples
            examples from one source at a time according to these probabilities.
        seed (`int`, optional, default None): The random seed used to choose a source for each example.
        stopping_strategy (`str`, defaults to `first_exhausted`):
            Two strategies are proposed right now.
            By default, `first_exhausted` is an undersampling strategy, i.e the dataset construction is stopped as soon as one dataset has ran out of samples.
            If the strategy is `all_exhausted`,  we use an oversampling strategy, i.e the dataset construction is stopped as soon as every samples of every dataset has been added at least once.
            Note that if the strategy is `all_exhausted`, the interleaved dataset size can get enormous:
            - with no probabilities, the resulting dataset will have max_length_datasets*nb_dataset samples.
            - with given probabilities, the resulting dataset will have more samples if some datasets have really low probability of visiting.

    Output:
        `datasets.IterableDataset`
    """
    datasets = [d._resolve_features() for d in datasets]
    _check_if_features_can_be_aligned([dset.features for dset in datasets])
    features = Features({k: v for features in _align_features([dset.features for dset in datasets]) for k, v in features.items()})
    ex_iterables = [d._ex_iterable for d in datasets]
    if probabilities is None:
        ex_iterable = CyclingMultiSourcesExamplesIterable(ex_iterables, stopping_strategy=stopping_strategy)
    else:
        generator = np.random.default_rng(seed)
        ex_iterable = RandomlyCyclingMultiSourcesExamplesIterable(ex_iterables, generator=generator, probabilities=probabilities, stopping_strategy=stopping_strategy)
    if info is None:
        info = DatasetInfo.from_merge([d.info for d in datasets])
    else:
        info = info.copy()
    info.features = features
    token_per_repo_id = {repo_id: token for dataset in datasets for repo_id, token in dataset._token_per_repo_id.items()}
    return IterableDataset(ex_iterable=ex_iterable, info=info, split=split, token_per_repo_id=token_per_repo_id)