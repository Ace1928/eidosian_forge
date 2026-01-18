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
class fz_stext_page(object):
    """
    A text page is a list of blocks, together with an overall
    bounding box.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    pool = property(_mupdf.fz_stext_page_pool_get, _mupdf.fz_stext_page_pool_set)
    mediabox = property(_mupdf.fz_stext_page_mediabox_get, _mupdf.fz_stext_page_mediabox_set)
    first_block = property(_mupdf.fz_stext_page_first_block_get, _mupdf.fz_stext_page_first_block_set)
    last_block = property(_mupdf.fz_stext_page_last_block_get, _mupdf.fz_stext_page_last_block_set)

    def __init__(self):
        _mupdf.fz_stext_page_swiginit(self, _mupdf.new_fz_stext_page())
    __swig_destroy__ = _mupdf.delete_fz_stext_page