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
def add_faiss_index_from_external_arrays(self, external_arrays: np.array, index_name: str, device: Optional[int]=None, string_factory: Optional[str]=None, metric_type: Optional[int]=None, custom_index: Optional['faiss.Index']=None, batch_size: int=1000, train_size: Optional[int]=None, faiss_verbose: bool=False, dtype=np.float32):
    """Add a dense index using Faiss for fast retrieval.
        The index is created using the vectors of `external_arrays`.
        You can specify `device` if you want to run it on GPU (`device` must be the GPU index).
        You can find more information about Faiss here:

        - For [string factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory)

        Args:
            external_arrays (`np.array`):
                If you want to use arrays from outside the lib for the index, you can set `external_arrays`.
                It will use `external_arrays` to create the Faiss index instead of the arrays in the given `column`.
            index_name (`str`):
                The `index_name`/identifier of the index.
                This is the `index_name` that is used to call [`~datasets.Dataset.get_nearest_examples`] or [`~datasets.Dataset.search`].
            device (Optional `Union[int, List[int]]`, *optional*):
                If positive integer, this is the index of the GPU to use. If negative integer, use all GPUs.
                If a list of positive integers is passed in, run only on those GPUs. By default it uses the CPU.
            string_factory (`str`, *optional*):
                This is passed to the index factory of Faiss to create the index.
                Default index class is `IndexFlat`.
            metric_type (`int`, *optional*):
                Type of metric. Ex: `faiss.faiss.METRIC_INNER_PRODUCT` or `faiss.METRIC_L2`.
            custom_index (`faiss.Index`, *optional*):
                Custom Faiss index that you already have instantiated and configured for your needs.
            batch_size (`int`, *optional*):
                Size of the batch to use while adding vectors to the FaissIndex. Default value is 1000.
                <Added version="2.4.0"/>
            train_size (`int`, *optional*):
                If the index needs a training step, specifies how many vectors will be used to train the index.
            faiss_verbose (`bool`, defaults to False):
                Enable the verbosity of the Faiss index.
            dtype (`numpy.dtype`):
                The dtype of the numpy arrays that are indexed. Default is np.float32.
        """
    super().add_faiss_index_from_external_arrays(external_arrays=external_arrays.astype(dtype), index_name=index_name, device=device, string_factory=string_factory, metric_type=metric_type, custom_index=custom_index, batch_size=batch_size, train_size=train_size, faiss_verbose=faiss_verbose)