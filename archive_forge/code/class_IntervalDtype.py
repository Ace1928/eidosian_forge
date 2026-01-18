from __future__ import annotations
from datetime import (
from decimal import Decimal
import re
from typing import (
import warnings
import numpy as np
import pytz
from pandas._libs import (
from pandas._libs.interval import Interval
from pandas._libs.properties import cache_readonly
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._libs.tslibs.offsets import BDay
from pandas.compat import pa_version_under10p1
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.util import capitalize_first_letter
@register_extension_dtype
class IntervalDtype(PandasExtensionDtype):
    """
    An ExtensionDtype for Interval data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    subtype : str, np.dtype
        The dtype of the Interval bounds.

    Attributes
    ----------
    subtype

    Methods
    -------
    None

    Examples
    --------
    >>> pd.IntervalDtype(subtype='int64', closed='both')
    interval[int64, both]
    """
    name = 'interval'
    kind: str_type = 'O'
    str = '|O08'
    base = np.dtype('O')
    num = 103
    _metadata = ('subtype', 'closed')
    _match = re.compile('(I|i)nterval\\[(?P<subtype>[^,]+(\\[.+\\])?)(, (?P<closed>(right|left|both|neither)))?\\]')
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}
    _subtype: None | np.dtype
    _closed: IntervalClosedType | None

    def __init__(self, subtype=None, closed: IntervalClosedType | None=None) -> None:
        from pandas.core.dtypes.common import is_string_dtype, pandas_dtype
        if closed is not None and closed not in {'right', 'left', 'both', 'neither'}:
            raise ValueError("closed must be one of 'right', 'left', 'both', 'neither'")
        if isinstance(subtype, IntervalDtype):
            if closed is not None and closed != subtype.closed:
                raise ValueError("dtype.closed and 'closed' do not match. Try IntervalDtype(dtype.subtype, closed) instead.")
            self._subtype = subtype._subtype
            self._closed = subtype._closed
        elif subtype is None:
            self._subtype = None
            self._closed = closed
        elif isinstance(subtype, str) and subtype.lower() == 'interval':
            self._subtype = None
            self._closed = closed
        else:
            if isinstance(subtype, str):
                m = IntervalDtype._match.search(subtype)
                if m is not None:
                    gd = m.groupdict()
                    subtype = gd['subtype']
                    if gd.get('closed', None) is not None:
                        if closed is not None:
                            if closed != gd['closed']:
                                raise ValueError("'closed' keyword does not match value specified in dtype string")
                        closed = gd['closed']
            try:
                subtype = pandas_dtype(subtype)
            except TypeError as err:
                raise TypeError('could not construct IntervalDtype') from err
            if CategoricalDtype.is_dtype(subtype) or is_string_dtype(subtype):
                msg = 'category, object, and string subtypes are not supported for IntervalDtype'
                raise TypeError(msg)
            self._subtype = subtype
            self._closed = closed

    @cache_readonly
    def _can_hold_na(self) -> bool:
        subtype = self._subtype
        if subtype is None:
            raise NotImplementedError('_can_hold_na is not defined for partially-initialized IntervalDtype')
        if subtype.kind in 'iu':
            return False
        return True

    @property
    def closed(self) -> IntervalClosedType:
        return self._closed

    @property
    def subtype(self):
        """
        The dtype of the Interval bounds.

        Examples
        --------
        >>> dtype = pd.IntervalDtype(subtype='int64', closed='both')
        >>> dtype.subtype
        dtype('int64')
        """
        return self._subtype

    @classmethod
    def construct_array_type(cls) -> type[IntervalArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import IntervalArray
        return IntervalArray

    @classmethod
    def construct_from_string(cls, string: str_type) -> IntervalDtype:
        """
        attempt to construct this type from a string, raise a TypeError
        if its not possible
        """
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        if string.lower() == 'interval' or cls._match.search(string) is not None:
            return cls(string)
        msg = f"Cannot construct a 'IntervalDtype' from '{string}'.\n\nIncorrectly formatted string passed to constructor. Valid formats include Interval or Interval[dtype] where dtype is numeric, datetime, or timedelta"
        raise TypeError(msg)

    @property
    def type(self) -> type[Interval]:
        return Interval

    def __str__(self) -> str_type:
        if self.subtype is None:
            return 'interval'
        if self.closed is None:
            return f'interval[{self.subtype}]'
        return f'interval[{self.subtype}, {self.closed}]'

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other.lower() in (self.name.lower(), str(self).lower())
        elif not isinstance(other, IntervalDtype):
            return False
        elif self.subtype is None or other.subtype is None:
            return True
        elif self.closed != other.closed:
            return False
        else:
            return self.subtype == other.subtype

    def __setstate__(self, state) -> None:
        self._subtype = state['subtype']
        self._closed = state.pop('closed', None)

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Return a boolean if we if the passed type is an actual dtype that we
        can match (via string or type)
        """
        if isinstance(dtype, str):
            if dtype.lower().startswith('interval'):
                try:
                    return cls.construct_from_string(dtype) is not None
                except (ValueError, TypeError):
                    return False
            else:
                return False
        return super().is_dtype(dtype)

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> IntervalArray:
        """
        Construct IntervalArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow
        from pandas.core.arrays import IntervalArray
        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks
        results = []
        for arr in chunks:
            if isinstance(arr, pyarrow.ExtensionArray):
                arr = arr.storage
            left = np.asarray(arr.field('left'), dtype=self.subtype)
            right = np.asarray(arr.field('right'), dtype=self.subtype)
            iarr = IntervalArray.from_arrays(left, right, closed=self.closed)
            results.append(iarr)
        if not results:
            return IntervalArray.from_arrays(np.array([], dtype=self.subtype), np.array([], dtype=self.subtype), closed=self.closed)
        return IntervalArray._concat_same_type(results)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        if not all((isinstance(x, IntervalDtype) for x in dtypes)):
            return None
        closed = cast('IntervalDtype', dtypes[0]).closed
        if not all((cast('IntervalDtype', x).closed == closed for x in dtypes)):
            return np.dtype(object)
        from pandas.core.dtypes.cast import find_common_type
        common = find_common_type([cast('IntervalDtype', x).subtype for x in dtypes])
        if common == object:
            return np.dtype(object)
        return IntervalDtype(common, closed=closed)

    @cache_readonly
    def index_class(self) -> type_t[IntervalIndex]:
        from pandas import IntervalIndex
        return IntervalIndex