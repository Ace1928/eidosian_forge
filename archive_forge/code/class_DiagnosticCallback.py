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
class DiagnosticCallback(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, description):
        if self.__class__ == DiagnosticCallback:
            _self = None
        else:
            _self = self
        _mupdf.DiagnosticCallback_swiginit(self, _mupdf.new_DiagnosticCallback(_self, description))

    def _print(self, message):
        return _mupdf.DiagnosticCallback__print(self, message)
    __swig_destroy__ = _mupdf.delete_DiagnosticCallback

    @staticmethod
    def s_print(self0, message):
        return _mupdf.DiagnosticCallback_s_print(self0, message)
    m_description = property(_mupdf.DiagnosticCallback_m_description_get, _mupdf.DiagnosticCallback_m_description_set)

    def __disown__(self):
        self.this.disown()
        _mupdf.disown_DiagnosticCallback(self)
        return weakref.proxy(self)