import abc
import enum
import inspect
import logging
from typing import Tuple
import typing
import warnings
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from _cffi_backend import FFI  # type: ignore
class SexpCapsule(CapsuleBase):
    __slots__ = ('_cdata',)

    def __init__(self, cdata: FFI.CData):
        assert is_cdata_sexp(cdata)
        _preserve(cdata)
        self._cdata = cdata

    def __del__(self):
        try:
            _release(self._cdata)
        except Exception as e:
            if _release is None:
                pass
            else:
                raise e