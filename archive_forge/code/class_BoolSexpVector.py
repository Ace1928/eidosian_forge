import abc
import atexit
import contextlib
import contextvars
import csv
import enum
import functools
import inspect
import os
import math
import platform
import signal
import subprocess
import textwrap
import threading
import typing
import warnings
from typing import Union
from rpy2.rinterface_lib import openrlib
import rpy2.rinterface_lib._rinterface_capi as _rinterface
import rpy2.rinterface_lib.embedded as embedded
import rpy2.rinterface_lib.conversion as conversion
from rpy2.rinterface_lib.conversion import _cdata_res_to_rinterface
import rpy2.rinterface_lib.memorymanagement as memorymanagement
from rpy2.rinterface_lib import na_values
from rpy2.rinterface_lib.sexp import NULL
from rpy2.rinterface_lib.sexp import NULLType
import rpy2.rinterface_lib.bufferprotocol as bufferprotocol
from rpy2.rinterface_lib import sexp
from rpy2.rinterface_lib.sexp import CharSexp  # noqa: F401
from rpy2.rinterface_lib.sexp import RTYPES
from rpy2.rinterface_lib.sexp import SexpVector
from rpy2.rinterface_lib.sexp import StrSexpVector
from rpy2.rinterface_lib.sexp import Sexp
from rpy2.rinterface_lib.sexp import SexpEnvironment
from rpy2.rinterface_lib.sexp import unserialize  # noqa: F401
from rpy2.rinterface_lib.sexp import emptyenv
from rpy2.rinterface_lib.sexp import baseenv
from rpy2.rinterface_lib.sexp import globalenv
class BoolSexpVector(SexpVectorWithNumpyInterface):
    """Array of booleans.

    Note that R is internally storing booleans as integers to
    allow an additional "NA" value to represent missingness."""
    _R_TYPE = openrlib.rlib.LGLSXP
    _R_SIZEOF_ELT = _rinterface.ffi.sizeof('Rboolean')
    _NP_TYPESTR = '|i'
    _R_VECTOR_ELT = openrlib.LOGICAL_ELT
    _R_SET_VECTOR_ELT = openrlib.SET_LOGICAL_ELT
    _R_GET_PTR = staticmethod(openrlib.LOGICAL)

    @staticmethod
    def _CAST_IN(x):
        if x is None or x == openrlib.rlib.R_NaInt:
            return NA_Logical
        else:
            return bool(x)

    def __getitem__(self, i: Union[int, slice]) -> Union[bool, 'sexp.NALogicalType', 'BoolSexpVector']:
        res: Union[bool, 'sexp.NALogicalType', 'BoolSexpVector']
        cdata = self.__sexp__._cdata
        if isinstance(i, int):
            i_c = _rinterface._python_index_to_c(cdata, i)
            elt = openrlib.LOGICAL_ELT(cdata, i_c)
            res = na_values.NA_Logical if elt == NA_Logical else bool(elt)
        elif isinstance(i, slice):
            res = type(self).from_iterable([openrlib.LOGICAL_ELT(cdata, i_c) for i_c in range(*i.indices(len(self)))])
        else:
            raise TypeError('Indices must be integers or slices, not %s' % type(i))
        return res

    def __setitem__(self, i: Union[int, slice], value) -> None:
        cdata = self.__sexp__._cdata
        if isinstance(i, int):
            i_c = _rinterface._python_index_to_c(cdata, i)
            openrlib.SET_LOGICAL_ELT(cdata, i_c, int(value))
        elif isinstance(i, slice):
            for i_c, v in zip(range(*i.indices(len(self))), value):
                openrlib.SET_LOGICAL_ELT(cdata, i_c, int(v))
        else:
            raise TypeError('Indices must be integers or slices, not %s' % type(i))

    def memoryview(self) -> memoryview:
        return vector_memoryview(self, 'int', 'i')