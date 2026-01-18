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
class _MissingArgType(SexpSymbol, metaclass=sexp.SingletonABC):
    """Missing argument in a call to an R function.

    When evaluating a function, arguments that are not specified
    can take a default value when named, otherwise they will
    take a special value that indicates that they are missing."""

    def __init__(self):
        if embedded.isready():
            tmp = sexp.Sexp(_rinterface.UnmanagedSexpCapsule(openrlib.rlib.R_MissingArg))
        else:
            tmp = sexp.Sexp(_rinterface.UninitializedRCapsule(RTYPES.SYMSXP.value))
        super().__init__(tmp)

    def __bool__(self) -> bool:
        """This is always False."""
        return False

    @property
    def __sexp__(self) -> typing.Union['_rinterface.SexpCapsule', '_rinterface.UninitializedRCapsule']:
        return self._sexpobject

    @__sexp__.setter
    def __sexp__(self, value: typing.Union['_rinterface.SexpCapsule', '_rinterface.UninitializedRCapsule']) -> None:
        raise TypeError('The capsule for the R object cannot be modified.')