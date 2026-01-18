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
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype):
    """
    An ExtensionDtype for Period data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    freq : str or DateOffset
        The frequency of this PeriodDtype.

    Attributes
    ----------
    freq

    Methods
    -------
    None

    Examples
    --------
    >>> pd.PeriodDtype(freq='D')
    period[D]

    >>> pd.PeriodDtype(freq=pd.offsets.MonthEnd())
    period[M]
    """
    type: type[Period] = Period
    kind: str_type = 'O'
    str = '|O08'
    base = np.dtype('O')
    num = 102
    _metadata = ('freq',)
    _match = re.compile('(P|p)eriod\\[(?P<freq>.+)\\]')
    _cache_dtypes: dict[BaseOffset, int] = {}
    __hash__ = PeriodDtypeBase.__hash__
    _freq: BaseOffset
    _supports_2d = True
    _can_fast_transpose = True

    def __new__(cls, freq) -> PeriodDtype:
        """
        Parameters
        ----------
        freq : PeriodDtype, BaseOffset, or string
        """
        if isinstance(freq, PeriodDtype):
            return freq
        if not isinstance(freq, BaseOffset):
            freq = cls._parse_dtype_strict(freq)
        if isinstance(freq, BDay):
            warnings.warn("PeriodDtype[B] is deprecated and will be removed in a future version. Use a DatetimeIndex with freq='B' instead", FutureWarning, stacklevel=find_stack_level())
        try:
            dtype_code = cls._cache_dtypes[freq]
        except KeyError:
            dtype_code = freq._period_dtype_code
            cls._cache_dtypes[freq] = dtype_code
        u = PeriodDtypeBase.__new__(cls, dtype_code, freq.n)
        u._freq = freq
        return u

    def __reduce__(self) -> tuple[type_t[Self], tuple[str_type]]:
        return (type(self), (self.name,))

    @property
    def freq(self) -> BaseOffset:
        """
        The frequency object of this PeriodDtype.

        Examples
        --------
        >>> dtype = pd.PeriodDtype(freq='D')
        >>> dtype.freq
        <Day>
        """
        return self._freq

    @classmethod
    def _parse_dtype_strict(cls, freq: str_type) -> BaseOffset:
        if isinstance(freq, str):
            if freq.startswith(('Period[', 'period[')):
                m = cls._match.search(freq)
                if m is not None:
                    freq = m.group('freq')
            freq_offset = to_offset(freq, is_period=True)
            if freq_offset is not None:
                return freq_offset
        raise TypeError(f'PeriodDtype argument should be string or BaseOffset, got {type(freq).__name__}')

    @classmethod
    def construct_from_string(cls, string: str_type) -> PeriodDtype:
        """
        Strict construction from a string, raise a TypeError if not
        possible
        """
        if isinstance(string, str) and string.startswith(('period[', 'Period[')) or isinstance(string, BaseOffset):
            try:
                return cls(freq=string)
            except ValueError:
                pass
        if isinstance(string, str):
            msg = f"Cannot construct a 'PeriodDtype' from '{string}'"
        else:
            msg = f"'construct_from_string' expects a string, got {type(string)}"
        raise TypeError(msg)

    def __str__(self) -> str_type:
        return self.name

    @property
    def name(self) -> str_type:
        return f'period[{self._freqstr}]'

    @property
    def na_value(self) -> NaTType:
        return NaT

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other in [self.name, capitalize_first_letter(self.name)]
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Return a boolean if we if the passed type is an actual dtype that we
        can match (via string or type)
        """
        if isinstance(dtype, str):
            if dtype.startswith(('period[', 'Period[')):
                try:
                    return cls._parse_dtype_strict(dtype) is not None
                except ValueError:
                    return False
            else:
                return False
        return super().is_dtype(dtype)

    @classmethod
    def construct_array_type(cls) -> type_t[PeriodArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import PeriodArray
        return PeriodArray

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> PeriodArray:
        """
        Construct PeriodArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow
        from pandas.core.arrays import PeriodArray
        from pandas.core.arrays.arrow._arrow_utils import pyarrow_array_to_numpy_and_mask
        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks
        results = []
        for arr in chunks:
            data, mask = pyarrow_array_to_numpy_and_mask(arr, dtype=np.dtype(np.int64))
            parr = PeriodArray(data.copy(), dtype=self, copy=False)
            parr[~mask] = NaT
            results.append(parr)
        if not results:
            return PeriodArray(np.array([], dtype='int64'), dtype=self, copy=False)
        return PeriodArray._concat_same_type(results)

    @cache_readonly
    def index_class(self) -> type_t[PeriodIndex]:
        from pandas import PeriodIndex
        return PeriodIndex