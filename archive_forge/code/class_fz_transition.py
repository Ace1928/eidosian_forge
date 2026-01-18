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
class fz_transition(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    type = property(_mupdf.fz_transition_type_get, _mupdf.fz_transition_type_set)
    duration = property(_mupdf.fz_transition_duration_get, _mupdf.fz_transition_duration_set)
    vertical = property(_mupdf.fz_transition_vertical_get, _mupdf.fz_transition_vertical_set)
    outwards = property(_mupdf.fz_transition_outwards_get, _mupdf.fz_transition_outwards_set)
    direction = property(_mupdf.fz_transition_direction_get, _mupdf.fz_transition_direction_set)
    state0 = property(_mupdf.fz_transition_state0_get, _mupdf.fz_transition_state0_set)
    state1 = property(_mupdf.fz_transition_state1_get, _mupdf.fz_transition_state1_set)

    def __init__(self):
        _mupdf.fz_transition_swiginit(self, _mupdf.new_fz_transition())
    __swig_destroy__ = _mupdf.delete_fz_transition