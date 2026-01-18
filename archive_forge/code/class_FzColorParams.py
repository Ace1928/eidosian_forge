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
class FzColorParams(object):
    """ Wrapper class for struct `fz_color_params`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, *args):
        """
        *Overload 1:*
        Equivalent to fz_default_color_params.

        |

        *Overload 2:*
        We use default copy constructor and operator=.  Constructor using raw copy of pre-existing `::fz_color_params`.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::fz_color_params`.
        """
        _mupdf.FzColorParams_swiginit(self, _mupdf.new_FzColorParams(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.FzColorParams_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_FzColorParams
    ri = property(_mupdf.FzColorParams_ri_get, _mupdf.FzColorParams_ri_set)
    bp = property(_mupdf.FzColorParams_bp_get, _mupdf.FzColorParams_bp_set)
    op = property(_mupdf.FzColorParams_op_get, _mupdf.FzColorParams_op_set)
    opm = property(_mupdf.FzColorParams_opm_get, _mupdf.FzColorParams_opm_set)
    s_num_instances = property(_mupdf.FzColorParams_s_num_instances_get, _mupdf.FzColorParams_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzColorParams_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzColorParams___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzColorParams___ne__(self, rhs)