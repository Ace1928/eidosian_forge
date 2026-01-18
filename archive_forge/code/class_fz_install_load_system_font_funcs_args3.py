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
class fz_install_load_system_font_funcs_args3(FzInstallLoadSystemFontFuncsArgs2):
    """
    Class derived from Swig Director class
    fz_install_load_system_font_funcs_args2, to allow
    implementation of fz_install_load_system_font_funcs with
    Python callbacks.
    """

    def __init__(self, f=None, f_cjk=None, f_fallback=None):
        super().__init__()
        self.f3 = f
        self.f_cjk3 = f_cjk
        self.f_fallback3 = f_fallback
        self.use_virtual_f(True if f else False)
        self.use_virtual_f_cjk(True if f_cjk else False)
        self.use_virtual_f_fallback(True if f_fallback else False)

    def ret_font(self, font):
        if font is None:
            return None
        elif isinstance(font, FzFont):
            return ll_fz_keep_font(font.m_internal)
        elif isinstance(font, fz_font):
            return font
        else:
            assert 0, f'Expected FzFont or fz_font, but fz_install_load_system_font_funcs() callback returned type(font)={type(font)!r}'

    def f(self, ctx, name, bold, italic, needs_exact_metrics):
        font = self.f3(name, bold, italic, needs_exact_metrics)
        return self.ret_font(font)

    def f_cjk(self, ctx, name, ordering, serif):
        font = self.f_cjk3(name, ordering, serif)
        return self.ret_font(font)

    def f_fallback(self, ctx, script, language, serif, bold, italic):
        font = self.f_fallback3(script, language, serif, bold, italic)
        return self.ret_font(font)