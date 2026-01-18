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
class fz_write_story_position(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    element = property(_mupdf.fz_write_story_position_element_get, _mupdf.fz_write_story_position_element_set)
    page_num = property(_mupdf.fz_write_story_position_page_num_get, _mupdf.fz_write_story_position_page_num_set)

    def __init__(self):
        _mupdf.fz_write_story_position_swiginit(self, _mupdf.new_fz_write_story_position())
    __swig_destroy__ = _mupdf.delete_fz_write_story_position