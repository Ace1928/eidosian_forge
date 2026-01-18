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
class FzInstallLoadSystemFontFuncsArgs2(FzInstallLoadSystemFontFuncsArgs):
    """ Wrapper class for struct fz_install_load_system_font_funcs_args with virtual fns for each fnptr; this is for use as a SWIG Director class."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self):
        """ == Constructor."""
        if self.__class__ == FzInstallLoadSystemFontFuncsArgs2:
            _self = None
        else:
            _self = self
        _mupdf.FzInstallLoadSystemFontFuncsArgs2_swiginit(self, _mupdf.new_FzInstallLoadSystemFontFuncsArgs2(_self))

    def use_virtual_f(self, use=True):
        """
        These methods set the function pointers in *m_internal
        to point to internal callbacks that call our virtual methods.
        """
        return _mupdf.FzInstallLoadSystemFontFuncsArgs2_use_virtual_f(self, use)

    def use_virtual_f_cjk(self, use=True):
        return _mupdf.FzInstallLoadSystemFontFuncsArgs2_use_virtual_f_cjk(self, use)

    def use_virtual_f_fallback(self, use=True):
        return _mupdf.FzInstallLoadSystemFontFuncsArgs2_use_virtual_f_fallback(self, use)

    def f(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        """ Default virtual method implementations; these all throw an exception."""
        return _mupdf.FzInstallLoadSystemFontFuncsArgs2_f(self, arg_0, arg_1, arg_2, arg_3, arg_4)

    def f_cjk(self, arg_0, arg_1, arg_2, arg_3):
        return _mupdf.FzInstallLoadSystemFontFuncsArgs2_f_cjk(self, arg_0, arg_1, arg_2, arg_3)

    def f_fallback(self, arg_0, arg_1, arg_2, arg_3, arg_4, arg_5):
        return _mupdf.FzInstallLoadSystemFontFuncsArgs2_f_fallback(self, arg_0, arg_1, arg_2, arg_3, arg_4, arg_5)
    __swig_destroy__ = _mupdf.delete_FzInstallLoadSystemFontFuncsArgs2

    def __disown__(self):
        self.this.disown()
        _mupdf.disown_FzInstallLoadSystemFontFuncsArgs2(self)
        return weakref.proxy(self)