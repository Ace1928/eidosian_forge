import copy
import json
import re
import sys
from collections.abc import Iterable, Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import InitVar, dataclass, field, fields
from functools import reduce, wraps
from operator import mul
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
from typing import Sequence as Sequence_
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
import pyarrow_hotfix  # noqa: F401  # to fix vulnerability on pyarrow<14.0.1
from pandas.api.extensions import ExtensionArray as PandasExtensionArray
from pandas.api.extensions import ExtensionDtype as PandasExtensionDtype
from .. import config
from ..naming import camelcase_to_snakecase, snakecase_to_camelcase
from ..table import array_cast
from ..utils import logging
from ..utils.py_utils import asdict, first_non_null_value, zip_dict
from .audio import Audio
from .image import Image, encode_pil_image
from .translation import Translation, TranslationVariableLanguages
class ArrayExtensionArray(pa.ExtensionArray):

    def __array__(self):
        zero_copy_only = _is_zero_copy_only(self.storage.type, unnest=True)
        return self.to_numpy(zero_copy_only=zero_copy_only)

    def __getitem__(self, i):
        return self.storage[i]

    def to_numpy(self, zero_copy_only=True):
        storage: pa.ListArray = self.storage
        null_mask = storage.is_null().to_numpy(zero_copy_only=False)
        if self.type.shape[0] is not None:
            size = 1
            null_indices = np.arange(len(storage))[null_mask] - np.arange(np.sum(null_mask))
            for i in range(self.type.ndims):
                size *= self.type.shape[i]
                storage = storage.flatten()
            numpy_arr = storage.to_numpy(zero_copy_only=zero_copy_only)
            numpy_arr = numpy_arr.reshape(len(self) - len(null_indices), *self.type.shape)
            if len(null_indices):
                numpy_arr = np.insert(numpy_arr.astype(np.float64), null_indices, np.nan, axis=0)
        else:
            shape = self.type.shape
            ndims = self.type.ndims
            arrays = []
            first_dim_offsets = np.array([off.as_py() for off in storage.offsets])
            for i, is_null in enumerate(null_mask):
                if is_null:
                    arrays.append(np.nan)
                else:
                    storage_el = storage[i:i + 1]
                    first_dim = first_dim_offsets[i + 1] - first_dim_offsets[i]
                    for _ in range(ndims):
                        storage_el = storage_el.flatten()
                    numpy_arr = storage_el.to_numpy(zero_copy_only=zero_copy_only)
                    arrays.append(numpy_arr.reshape(first_dim, *shape[1:]))
            if len(np.unique(np.diff(first_dim_offsets))) > 1:
                numpy_arr = np.empty(len(arrays), dtype=object)
                numpy_arr[:] = arrays
            else:
                numpy_arr = np.array(arrays)
        return numpy_arr

    def to_pylist(self):
        zero_copy_only = _is_zero_copy_only(self.storage.type, unnest=True)
        numpy_arr = self.to_numpy(zero_copy_only=zero_copy_only)
        if self.type.shape[0] is None and numpy_arr.dtype == object:
            return [arr.tolist() for arr in numpy_arr.tolist()]
        else:
            return numpy_arr.tolist()