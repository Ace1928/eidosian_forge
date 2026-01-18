import contextlib
import copy
import fnmatch
import itertools
import json
import math
import os
import posixpath
import re
import shutil
import sys
import tempfile
import time
import warnings
import weakref
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from functools import partial, wraps
from io import BytesIO
from math import ceil, floor
from pathlib import Path
from random import sample
from typing import (
from typing import Sequence as Sequence_
import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from huggingface_hub import CommitInfo, CommitOperationAdd, CommitOperationDelete, DatasetCard, DatasetCardData, HfApi
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map
from . import config
from .arrow_reader import ArrowReader
from .arrow_writer import ArrowWriter, OptimizedTypedSequence
from .data_files import sanitize_patterns
from .download.streaming_download_manager import xgetsize
from .features import Audio, ClassLabel, Features, Image, Sequence, Value
from .features.features import (
from .filesystems import is_remote_filesystem
from .fingerprint import (
from .formatting import format_table, get_format_type_from_alias, get_formatter, query_table
from .formatting.formatting import LazyDict, _is_range_contiguous
from .info import DatasetInfo, DatasetInfosDict
from .naming import _split_re
from .search import IndexableMixin
from .splits import NamedSplit, Split, SplitDict, SplitInfo
from .table import (
from .tasks import TaskTemplate
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.deprecation_utils import deprecated
from .utils.file_utils import estimate_dataset_size
from .utils.hub import list_files_info, preupload_lfs_files
from .utils.info_utils import is_small_dataset
from .utils.metadata import MetadataConfigs
from .utils.py_utils import (
from .utils.stratify import stratified_shuffle_split_generate_indices
from .utils.tf_utils import dataset_to_tf, minimal_tf_collate_fn, multiprocess_dataset_to_tf
from .utils.typing import ListLike, PathLike
def _interleave_map_style_datasets(datasets: List['Dataset'], probabilities: Optional[List[float]]=None, seed: Optional[int]=None, info: Optional[DatasetInfo]=None, split: Optional[NamedSplit]=None, stopping_strategy: Literal['first_exhausted', 'all_exhausted']='first_exhausted', **kwargs) -> 'Dataset':
    """
    Interleave several map-style datasets (sources) into a single map-style dataset.
    The new dataset is constructed by alternating between the sources to get the examples.
    If `probabilities = None` (default) the new dataset is constructed by cycling between each source to get the examples.
    If `probabilities` is not `None, the new dataset is constructed by getting examples from a random source at a time according to the provided probabilities.

    Args:
        datasets (`List[Dataset]`): list of datasets to interleave
        probabilities (`List[float]`, optional, default None): If specified, the new dataset is constructed by sampling
            examples from one source at a time according to these probabilities.
        seed (`int`, optional, default None): The random seed used to choose a source for each example.
        info (:class:`DatasetInfo`, optional): Dataset information, like description, citation, etc.
        split (:class:`NamedSplit`, optional): Name of the dataset split.
        stopping_strategy (`str`, defaults to `first_exhausted`):
            Two strategies are proposed right now.
            By default, `first_exhausted` is an undersampling strategy, i.e the dataset construction is stopped as soon as one dataset has ran out of samples.
            If the strategy is `all_exhausted`,  we use an oversampling strategy, i.e the dataset construction is stopped as soon as every samples of every dataset has been added at least once.
            Note that if the strategy is `all_exhausted`, the interleaved dataset size can get enormous:
            - with no probabilities, the resulting dataset will have max_length_datasets*nb_dataset samples.
            - with given probabilities, the resulting dataset will have more samples if some datasets have really low probability of visiting.
        **kwargs (additional keyword arguments): Keyword arguments to be passed to :meth:`datasets.Datasets.select` when selecting the indices used to interleave the datasets.

    Output:
        :class:`datasets.Dataset`
    """
    if stopping_strategy not in ['first_exhausted', 'all_exhausted']:
        raise ValueError(f'{stopping_strategy} stopping strategy in `interleave_datasets` is not implemented yet with a list of {type(datasets[0])}')
    concatenated_datasets = _concatenate_map_style_datasets(datasets, info=info, split=split)
    lengths = [len(dset) for dset in datasets]
    offsets = np.cumsum([0] + lengths[:-1])
    oversampling = stopping_strategy == 'all_exhausted'
    if probabilities is None and (not oversampling):
        indices = (offsets.reshape(1, -1) + np.arange(min(lengths)).reshape(-1, 1)).flatten().tolist()
    elif probabilities is None:
        indices = np.mod(np.arange(max(lengths)).reshape(-1, 1), np.array(lengths).reshape(1, -1))
        indices = (indices + offsets).flatten().tolist()
    else:
        is_exhausted = np.full(len(lengths), False)
        bool_strategy_func = np.all if oversampling else np.any

        def iter_random_indices():
            """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
            rng = np.random.default_rng(seed)
            while True:
                yield from (int(i) for i in rng.choice(len(datasets), size=1000, p=probabilities))
        current_index = [0] * len(datasets)
        indices = []
        for source_idx in iter_random_indices():
            if bool_strategy_func(is_exhausted):
                break
            indices.append(current_index[source_idx] + offsets[source_idx])
            current_index[source_idx] += 1
            if current_index[source_idx] >= lengths[source_idx]:
                is_exhausted[source_idx] = True
                current_index[source_idx] = 0
    return concatenated_datasets.select(indices, **kwargs)