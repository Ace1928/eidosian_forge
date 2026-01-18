from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class PdfFilterOptions2(PdfFilterOptions):
    """ Wrapper class for struct pdf_filter_options with virtual fns for each fnptr; this is for use as a SWIG Director class."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self):
        """ == Constructor."""
        if self.__class__ == PdfFilterOptions2:
            _self = None
        else:
            _self = self
        _mupdf.PdfFilterOptions2_swiginit(self, _mupdf.new_PdfFilterOptions2(_self))

    def use_virtual_complete(self, use=True):
        """
        These methods set the function pointers in *m_internal
        to point to internal callbacks that call our virtual methods.
        """
        return _mupdf.PdfFilterOptions2_use_virtual_complete(self, use)

    def complete(self, arg_0, arg_1):
        """ Default virtual method implementations; these all throw an exception."""
        return _mupdf.PdfFilterOptions2_complete(self, arg_0, arg_1)
    __swig_destroy__ = _mupdf.delete_PdfFilterOptions2

    def __disown__(self):
        self.this.disown()
        _mupdf.disown_PdfFilterOptions2(self)
        return weakref.proxy(self)