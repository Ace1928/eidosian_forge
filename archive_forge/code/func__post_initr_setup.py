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
def _post_initr_setup() -> None:
    emptyenv.__sexp__ = _rinterface.SexpCapsule(openrlib.rlib.R_EmptyEnv)
    baseenv.__sexp__ = _rinterface.SexpCapsule(openrlib.rlib.R_BaseEnv)
    globalenv.__sexp__ = _rinterface.SexpCapsule(openrlib.rlib.R_GlobalEnv)
    NULL._sexpobject = _rinterface.UnmanagedSexpCapsule(openrlib.rlib.R_NilValue)
    MissingArg._sexpobject = _rinterface.UnmanagedSexpCapsule(openrlib.rlib.R_MissingArg)
    global NA_Character
    na_values.NA_Character = sexp.NACharacterType()
    NA_Character = na_values.NA_Character
    global NA_Integer
    na_values.NA_Integer = sexp.NAIntegerType(openrlib.rlib.R_NaInt)
    NA_Integer = na_values.NA_Integer
    global NA_Logical, NA
    na_values.NA_Logical = sexp.NALogicalType(openrlib.rlib.R_NaInt)
    NA_Logical = na_values.NA_Logical
    NA = NA_Logical
    global NA_Real
    na_values.NA_Real = sexp.NARealType(openrlib.rlib.R_NaReal)
    NA_Real = na_values.NA_Real
    global NA_Complex
    na_values.NA_Complex = sexp.NAComplexType(_rinterface.ffi.new('Rcomplex *', [openrlib.rlib.R_NaReal, openrlib.rlib.R_NaReal]))
    NA_Complex = na_values.NA_Complex
    warn_about_thread = False
    if threading.current_thread() is threading.main_thread():
        try:
            signal.signal(signal.SIGINT, _sigint_handler)
        except ValueError as ve:
            if str(ve) == 'signal only works in main thread':
                warn_about_thread = True
            else:
                raise ve
    else:
        warn_about_thread = True
    if warn_about_thread:
        warnings.warn(textwrap.dedent('R is not initialized by the main thread.\n                Its taking over SIGINT cannot be reversed here, and as a\n                consequence the embedded R cannot be interrupted with Ctrl-C.\n                Consider (re)setting the signal handler of your choice from\n                the main thread.'))
    _update_R_ENC_PY()