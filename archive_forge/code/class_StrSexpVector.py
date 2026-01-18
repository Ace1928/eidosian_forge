import abc
import collections.abc
from collections import OrderedDict
import enum
import itertools
import typing
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface_lib._rinterface_capi as _rinterface
from rpy2.rinterface_lib._rinterface_capi import _evaluated_promise
from rpy2.rinterface_lib._rinterface_capi import SupportsSEXP
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib.conversion import _cdata_res_to_rinterface
from rpy2.rinterface_lib import na_values
class StrSexpVector(SexpVector):
    """R vector of strings."""
    _R_TYPE = openrlib.rlib.STRSXP
    _R_GET_PTR = openrlib._STRING_PTR
    _R_SIZEOF_ELT = None
    _R_VECTOR_ELT = openrlib.rlib.STRING_ELT
    _R_SET_VECTOR_ELT = openrlib.rlib.SET_STRING_ELT
    _CAST_IN = _as_charsxp_cdata

    def __getitem__(self, i: typing.Union[int, slice]) -> typing.Union['StrSexpVector', str, 'NACharacterType']:
        cdata = self.__sexp__._cdata
        res: typing.Union['StrSexpVector', str, 'NACharacterType']
        if isinstance(i, int):
            i_c = _rinterface._python_index_to_c(cdata, i)
            _ = _rinterface._string_getitem(cdata, i_c)
            if _ is None:
                res = na_values.NA_Character
            else:
                res = _
        elif isinstance(i, slice):
            res = self.from_iterable([_rinterface._string_getitem(cdata, i_c) for i_c in range(*i.indices(len(self)))])
        else:
            raise TypeError('Indices must be integers or slices, not %s' % type(i))
        return res

    def __setitem__(self, i: typing.Union[int, slice], value: typing.Union[str, typing.Sequence[typing.Optional[str]], 'StrSexpVector', 'NACharacterType']) -> None:
        cdata = self.__sexp__._cdata
        if isinstance(i, int):
            i_c = _rinterface._python_index_to_c(cdata, i)
            if isinstance(value, Sexp):
                val_cdata = value.__sexp__._cdata
            else:
                if not isinstance(value, str):
                    value = str(value)
                val_cdata = _as_charsxp_cdata(value)
            self._R_SET_VECTOR_ELT(cdata, i_c, val_cdata)
        elif isinstance(i, slice):
            value_slice: typing.Iterable
            if isinstance(value, NACharacterType) or isinstance(value, str):
                value_slice = itertools.cycle((value,))
            elif len(value) == 1:
                value_slice = itertools.cycle(value)
            else:
                value_slice = value
            for i_c, _ in zip(range(*i.indices(len(self))), value_slice):
                if _ is None:
                    v_cdata = openrlib.rlib.R_NaString
                else:
                    if isinstance(_, str):
                        v = _
                    else:
                        v = str(_)
                    v_cdata = _as_charsxp_cdata(v)
                self._R_SET_VECTOR_ELT(cdata, i_c, v_cdata)
        else:
            raise TypeError('Indices must be integers or slices, not %s' % type(i))

    def get_charsxp(self, i: int) -> CharSexp:
        """Get the R CharSexp objects for the index i."""
        i_c = _rinterface._python_index_to_c(self.__sexp__._cdata, i)
        return CharSexp(_rinterface.SexpCapsule(openrlib.rlib.STRING_ELT(self.__sexp__._cdata, i_c)))