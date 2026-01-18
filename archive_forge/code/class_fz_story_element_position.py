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
class fz_story_element_position(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    depth = property(_mupdf.fz_story_element_position_depth_get, _mupdf.fz_story_element_position_depth_set)
    heading = property(_mupdf.fz_story_element_position_heading_get, _mupdf.fz_story_element_position_heading_set)
    id = property(_mupdf.fz_story_element_position_id_get, _mupdf.fz_story_element_position_id_set)
    href = property(_mupdf.fz_story_element_position_href_get, _mupdf.fz_story_element_position_href_set)
    rect = property(_mupdf.fz_story_element_position_rect_get, _mupdf.fz_story_element_position_rect_set)
    text = property(_mupdf.fz_story_element_position_text_get, _mupdf.fz_story_element_position_text_set)
    open_close = property(_mupdf.fz_story_element_position_open_close_get, _mupdf.fz_story_element_position_open_close_set)
    rectangle_num = property(_mupdf.fz_story_element_position_rectangle_num_get, _mupdf.fz_story_element_position_rectangle_num_set)

    def __init__(self):
        _mupdf.fz_story_element_position_swiginit(self, _mupdf.new_fz_story_element_position())
    __swig_destroy__ = _mupdf.delete_fz_story_element_position