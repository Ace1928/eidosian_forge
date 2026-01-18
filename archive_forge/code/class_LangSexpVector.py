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
class LangSexpVector(SexpVector):
    """An R language object.

    To create from a (Python) string containing R code
    use the classmethod `from_string`."""
    _R_TYPE = openrlib.rlib.LANGSXP
    _R_GET_PTR = None
    _CAST_IN = None
    _R_SIZEOF_ELT = None
    _R_VECTOR_ELT = None
    _R_SET_VECTOR_ELT = None

    @_cdata_res_to_rinterface
    def __getitem__(self, i: int):
        cdata = self.__sexp__._cdata
        i_c = _rinterface._python_index_to_c(cdata, i)
        return openrlib.rlib.CAR(openrlib.rlib.Rf_nthcdr(cdata, i_c))

    def __setitem__(self, i: typing.Union[int, slice], value: sexp.SupportsSEXP) -> None:
        if isinstance(i, slice):
            raise NotImplementedError('Assigning slices to LangSexpVectors is not yet implemented.')
        cdata = self.__sexp__._cdata
        i_c = _rinterface._python_index_to_c(cdata, i)
        openrlib.rlib.SETCAR(openrlib.rlib.Rf_nthcdr(cdata, i_c), value.__sexp__._cdata)

    @classmethod
    def from_string(cls: typing.Type[LangSexpVector_VT], s: str) -> LangSexpVector_VT:
        """Create an R language object from a string.

        This creates an unevaluated R language object.

        Args:
            s: R source code in a string.

        Returns:
            An instance of the class the method is called from.
        """
        return cls(_get_str2lang()(s))