from __future__ import annotations
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType
from pandas.core.dtypes.common import _get_dtype, is_string_dtype
from pyarrow.types import is_dictionary
from modin.pandas.indexing import is_range_like
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
class ColNameCodec:
    IDX_COL_NAME = '__index__'
    ROWID_COL_NAME = '__rowid__'
    UNNAMED_IDX_COL_NAME = '__index__0__N'
    _IDX_NAME_PATTERN = re.compile(f'{IDX_COL_NAME}\\d+_(.*)')
    _RESERVED_NAMES = (MODIN_UNNAMED_SERIES_LABEL, ROWID_COL_NAME)
    _COL_TYPES = Union[str, int, float, pandas.Timestamp, None]
    _COL_NAME_TYPE = Union[_COL_TYPES, Tuple[_COL_TYPES, ...]]

    def _encode_tuple(values: Tuple[_COL_TYPES, ...]) -> str:
        dst = ['_T']
        count = len(values)
        for value in values:
            if isinstance(value, str):
                dst.append(value.replace('_', '_Q'))
            else:
                dst.append(ColNameCodec._ENCODERS[type(value)](value))
            count -= 1
            if count != 0:
                dst.append('_T')
        return ''.join(dst)

    def _decode_tuple(encoded: str) -> Tuple[_COL_TYPES, ...]:
        items = []
        for item in encoded[2:].split('_T'):
            dec = None if len(item) < 2 or item[0] != '_' else ColNameCodec._DECODERS.get(item[1], None)
            items.append(item.replace('_Q', '_') if dec is None else dec(item))
        return tuple(items)
    _ENCODERS = {tuple: _encode_tuple, type(None): lambda v: '_N', str: lambda v: '_E' if len(v) == 0 else '_S' + v[1:] if v[0] == '_' else v, int: lambda v: f'_I{v}', float: lambda v: f'_F{v}', pandas.Timestamp: lambda v: f'_D{v.timestamp()}_{v.tz}'}
    _DECODERS = {'T': _decode_tuple, 'N': lambda v: None, 'E': lambda v: '', 'S': lambda v: '_' + v[2:], 'I': lambda v: int(v[2:]), 'F': lambda v: float(v[2:]), 'D': lambda v: pandas.Timestamp.fromtimestamp(float(v[2:(idx := v.index('_', 2))]), tz=v[idx + 1:])}

    @staticmethod
    @lru_cache(1024)
    def encode(name: _COL_NAME_TYPE, ignore_reserved: bool=True) -> str:
        """
        Encode column name.

        The supported name types are specified in the type hints. Non-string names
        are converted to string and prefixed with a corresponding tag.

        Parameters
        ----------
        name : str, int, float, Timestamp, None, tuple
            Column name to be encoded.
        ignore_reserved : bool, default: True
            Do not encode reserved names.

        Returns
        -------
        str
            Encoded name.
        """
        if ignore_reserved and isinstance(name, str) and (name.startswith(ColNameCodec.IDX_COL_NAME) or name in ColNameCodec._RESERVED_NAMES):
            return name
        try:
            return ColNameCodec._ENCODERS[type(name)](name)
        except KeyError:
            raise TypeError(f'Unsupported column name: {name}')

    @staticmethod
    @lru_cache(1024)
    def decode(name: str) -> _COL_NAME_TYPE:
        """
        Decode column name, previously encoded with encode_col_name().

        Parameters
        ----------
        name : str
            Encoded name.

        Returns
        -------
        str, int, float, Timestamp, None, tuple
            Decoded name.
        """
        if len(name) < 2 or name[0] != '_' or name.startswith(ColNameCodec.IDX_COL_NAME) or (name in ColNameCodec._RESERVED_NAMES):
            return name
        try:
            return ColNameCodec._DECODERS[name[1]](name)
        except KeyError:
            raise ValueError(f'Invalid encoded column name: {name}')

    @staticmethod
    def mangle_index_names(names: List[_COL_NAME_TYPE]) -> List[str]:
        """
        Return mangled index names for index labels.

        Mangled names are used for index columns because index
        labels cannot always be used as HDK table column
        names. E.e. label can be a non-string value or an
        unallowed string (empty strings, etc.) for a table column
        name.

        Parameters
        ----------
        names : list of str
            Index labels.

        Returns
        -------
        list of str
            Mangled names.
        """
        pref = ColNameCodec.IDX_COL_NAME
        return [f'{pref}{i}_{ColNameCodec.encode(n)}' for i, n in enumerate(names)]

    @staticmethod
    def demangle_index_names(cols: List[str]) -> Union[_COL_NAME_TYPE, List[_COL_NAME_TYPE]]:
        """
        Demangle index column names to index labels.

        Parameters
        ----------
        cols : list of str
            Index column names.

        Returns
        -------
        list or a single demangled name
            Demangled index names.
        """
        if len(cols) == 1:
            return ColNameCodec.demangle_index_name(cols[0])
        return [ColNameCodec.demangle_index_name(n) for n in cols]

    @staticmethod
    def demangle_index_name(col: str) -> _COL_NAME_TYPE:
        """
        Demangle index column name into index label.

        Parameters
        ----------
        col : str
            Index column name.

        Returns
        -------
        str
            Demangled index name.
        """
        match = ColNameCodec._IDX_NAME_PATTERN.search(col)
        if match:
            name = match.group(1)
            if name == MODIN_UNNAMED_SERIES_LABEL:
                return None
            return ColNameCodec.decode(name)
        return col

    @staticmethod
    def concat_index_names(frames) -> Dict[str, Any]:
        """
        Calculate the index names and dtypes.

        Calculate the index names and dtypes, that the index
        columns will have after the frames concatenation.

        Parameters
        ----------
        frames : list[HdkOnNativeDataframe]

        Returns
        -------
        Dict[str, Any]
        """
        first = frames[0]
        names = {}
        if first._index_width() > 1:
            dtypes = first._dtypes
            for n in first._index_cols:
                names[n] = dtypes[n]
        else:
            mangle = ColNameCodec.mangle_index_names
            idx_names = set()
            for f in frames:
                if f._index_cols is not None:
                    idx_names.update(f._index_cols)
                elif f.has_index_cache:
                    idx_names.update(mangle(f.index.names))
                else:
                    idx_names.add(ColNameCodec.UNNAMED_IDX_COL_NAME)
                if len(idx_names) > 1:
                    idx_names = [ColNameCodec.UNNAMED_IDX_COL_NAME]
                    break
            name = next(iter(idx_names))
            if first._index_cols is not None:
                names[name] = first._dtypes.iloc[0]
            elif first.has_index_cache:
                names[name] = first.index.dtype
            else:
                names[name] = _get_dtype(int)
        return names