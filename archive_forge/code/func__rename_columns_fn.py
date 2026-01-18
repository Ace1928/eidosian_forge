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
def _rename_columns_fn(example: Dict, column_mapping: Dict[str, str]):
    if any((col not in example for col in column_mapping)):
        raise ValueError(f'Error when renaming {list(column_mapping)} to {list(column_mapping.values())}: columns {set(column_mapping) - set(example)} are not in the dataset.')
    if any((col in example for col in column_mapping.values())):
        raise ValueError(f'Error when renaming {list(column_mapping)} to {list(column_mapping.values())}: columns {set(example) - set(column_mapping.values())} are already in the dataset.')
    return {new_column_name: example[original_column_name] for original_column_name, new_column_name in column_mapping.items()}