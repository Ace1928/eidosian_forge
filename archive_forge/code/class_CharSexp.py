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
class CharSexp(Sexp):
    """R's internal (C API-level) scalar for strings."""
    _R_TYPE = openrlib.rlib.CHARSXP
    _NCHAR_MSG = openrlib.ffi.new('char []', b'rpy2.rinterface.CharSexp.nchar')

    @property
    def encoding(self) -> CETYPE:
        return CETYPE(openrlib.rlib.Rf_getCharCE(self.__sexp__._cdata))

    def nchar(self, what: NCHAR_TYPE=NCHAR_TYPE.Bytes) -> int:
        return openrlib.rlib.R_nchar(self.__sexp__._cdata, what.value, openrlib.rlib.FALSE, openrlib.rlib.FALSE, self._NCHAR_MSG)