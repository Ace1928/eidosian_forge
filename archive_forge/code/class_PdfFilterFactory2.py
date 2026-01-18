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
class PdfFilterFactory2(PdfFilterFactory):
    """ Wrapper class for struct pdf_filter_factory with virtual fns for each fnptr; this is for use as a SWIG Director class."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self):
        """ == Constructor."""
        if self.__class__ == PdfFilterFactory2:
            _self = None
        else:
            _self = self
        _mupdf.PdfFilterFactory2_swiginit(self, _mupdf.new_PdfFilterFactory2(_self))

    def use_virtual_filter(self, use=True):
        """
        These methods set the function pointers in *m_internal
        to point to internal callbacks that call our virtual methods.
        """
        return _mupdf.PdfFilterFactory2_use_virtual_filter(self, use)

    def filter(self, arg_0, arg_1, arg_2, arg_3, arg_4, arg_5):
        """ Default virtual method implementations; these all throw an exception."""
        return _mupdf.PdfFilterFactory2_filter(self, arg_0, arg_1, arg_2, arg_3, arg_4, arg_5)
    __swig_destroy__ = _mupdf.delete_PdfFilterFactory2

    def __disown__(self):
        self.this.disown()
        _mupdf.disown_PdfFilterFactory2(self)
        return weakref.proxy(self)