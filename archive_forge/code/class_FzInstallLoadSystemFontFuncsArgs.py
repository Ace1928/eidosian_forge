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
class FzInstallLoadSystemFontFuncsArgs(object):
    """
     Wrapper class for struct `fz_install_load_system_font_funcs_args`.
    Extra struct containing fz_install_load_system_font_funcs()'s args,
    which we wrap with virtual_fnptrs set to allow use from Python/C# via
    Swig Directors.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_install_load_system_font_funcs2(self):
        """
        Class-aware wrapper for `::fz_install_load_system_font_funcs2()`.
        Alternative to fz_install_load_system_font_funcs() that takes args in a
        struct, to allow use from Python/C# via Swig Directors.
        """
        return _mupdf.FzInstallLoadSystemFontFuncsArgs_fz_install_load_system_font_funcs2(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_install_load_system_font_funcs_args`.
        """
        _mupdf.FzInstallLoadSystemFontFuncsArgs_swiginit(self, _mupdf.new_FzInstallLoadSystemFontFuncsArgs(*args))
    __swig_destroy__ = _mupdf.delete_FzInstallLoadSystemFontFuncsArgs
    m_internal = property(_mupdf.FzInstallLoadSystemFontFuncsArgs_m_internal_get, _mupdf.FzInstallLoadSystemFontFuncsArgs_m_internal_set)
    s_num_instances = property(_mupdf.FzInstallLoadSystemFontFuncsArgs_s_num_instances_get, _mupdf.FzInstallLoadSystemFontFuncsArgs_s_num_instances_set, doc=' Wrapped data is held by value.')

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzInstallLoadSystemFontFuncsArgs_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzInstallLoadSystemFontFuncsArgs___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzInstallLoadSystemFontFuncsArgs___ne__(self, rhs)