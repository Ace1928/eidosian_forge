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
class pdf_layer_config_ui(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    text = property(_mupdf.pdf_layer_config_ui_text_get, _mupdf.pdf_layer_config_ui_text_set)
    depth = property(_mupdf.pdf_layer_config_ui_depth_get, _mupdf.pdf_layer_config_ui_depth_set)
    type = property(_mupdf.pdf_layer_config_ui_type_get, _mupdf.pdf_layer_config_ui_type_set)
    selected = property(_mupdf.pdf_layer_config_ui_selected_get, _mupdf.pdf_layer_config_ui_selected_set)
    locked = property(_mupdf.pdf_layer_config_ui_locked_get, _mupdf.pdf_layer_config_ui_locked_set)

    def __init__(self):
        _mupdf.pdf_layer_config_ui_swiginit(self, _mupdf.new_pdf_layer_config_ui())
    __swig_destroy__ = _mupdf.delete_pdf_layer_config_ui