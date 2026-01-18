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
class PandasArrayExtensionArray(PandasExtensionArray):

    def __init__(self, data: np.ndarray, copy: bool=False):
        self._data = data if not copy else np.array(data)
        self._dtype = PandasArrayExtensionDtype(data.dtype)

    def __array__(self, dtype=None):
        """
        Convert to NumPy Array.
        Note that Pandas expects a 1D array when dtype is set to object.
        But for other dtypes, the returned shape is the same as the one of ``data``.

        More info about pandas 1D requirement for PandasExtensionArray here:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.extensions.ExtensionArray.html#pandas.api.extensions.ExtensionArray

        """
        if dtype == object:
            out = np.empty(len(self._data), dtype=object)
            for i in range(len(self._data)):
                out[i] = self._data[i]
            return out
        if dtype is None:
            return self._data
        else:
            return self._data.astype(dtype)

    def copy(self, deep: bool=False) -> 'PandasArrayExtensionArray':
        return PandasArrayExtensionArray(self._data, copy=True)

    @classmethod
    def _from_sequence(cls, scalars, dtype: Optional[PandasArrayExtensionDtype]=None, copy: bool=False) -> 'PandasArrayExtensionArray':
        if len(scalars) > 1 and all((isinstance(x, np.ndarray) and x.shape == scalars[0].shape and (x.dtype == scalars[0].dtype) for x in scalars)):
            data = np.array(scalars, dtype=dtype if dtype is None else dtype.value_type, copy=copy)
        else:
            data = np.empty(len(scalars), dtype=object)
            data[:] = scalars
        return cls(data, copy=copy)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence_['PandasArrayExtensionArray']) -> 'PandasArrayExtensionArray':
        if len(to_concat) > 1 and all((va._data.shape == to_concat[0]._data.shape and va._data.dtype == to_concat[0]._data.dtype for va in to_concat)):
            data = np.vstack([va._data for va in to_concat])
        else:
            data = np.empty(len(to_concat), dtype=object)
            data[:] = [va._data for va in to_concat]
        return cls(data, copy=False)

    @property
    def dtype(self) -> PandasArrayExtensionDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    def isna(self) -> np.ndarray:
        return np.array([pd.isna(arr).any() for arr in self._data])

    def __setitem__(self, key: Union[int, slice, np.ndarray], value: Any) -> None:
        raise NotImplementedError()

    def __getitem__(self, item: Union[int, slice, np.ndarray]) -> Union[np.ndarray, 'PandasArrayExtensionArray']:
        if isinstance(item, int):
            return self._data[item]
        return PandasArrayExtensionArray(self._data[item], copy=False)

    def take(self, indices: Sequence_[int], allow_fill: bool=False, fill_value: bool=None) -> 'PandasArrayExtensionArray':
        indices: np.ndarray = np.asarray(indices, dtype=int)
        if allow_fill:
            fill_value = self.dtype.na_value if fill_value is None else np.asarray(fill_value, dtype=self.dtype.value_type)
            mask = indices == -1
            if (indices < -1).any():
                raise ValueError('Invalid value in `indices`, must be all >= -1 for `allow_fill` is True')
            elif len(self) > 0:
                pass
            elif not np.all(mask):
                raise IndexError('Invalid take for empty PandasArrayExtensionArray, must be all -1.')
            else:
                data = np.array([fill_value] * len(indices), dtype=self.dtype.value_type)
                return PandasArrayExtensionArray(data, copy=False)
        took = self._data.take(indices, axis=0)
        if allow_fill and mask.any():
            took[mask] = [fill_value] * np.sum(mask)
        return PandasArrayExtensionArray(took, copy=False)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> np.ndarray:
        if not isinstance(other, PandasArrayExtensionArray):
            raise NotImplementedError(f'Invalid type to compare to: {type(other)}')
        return (self._data == other._data).all()