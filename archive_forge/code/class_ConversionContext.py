import contextvars
from functools import singledispatch
import os
from typing import Any
from typing import Optional
import typing
import warnings
from rpy2.rinterface_lib import _rinterface_capi
import rpy2.rinterface_lib.sexp
import rpy2.rinterface_lib.conversion
import rpy2.rinterface
class ConversionContext(object):
    """
    Context manager for instances of class Converter.
    """

    def __init__(self, ctx_converter):
        assert isinstance(ctx_converter, Converter)
        self._original_converter = converter_ctx.get()
        self.ctx_converter = Converter('Converter-%i-in-context' % id(self), template=ctx_converter)

    def __enter__(self):
        set_conversion(self.ctx_converter)
        return self.ctx_converter

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_conversion(self._original_converter)
        return False