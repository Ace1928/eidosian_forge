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
class fz_location(object):
    """
    Locations within the document are referred to in terms of
    chapter and page, rather than just a page number. For some
    documents (such as epub documents with large numbers of pages
    broken into many chapters) this can make navigation much faster
    as only the required chapter needs to be decoded at a time.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    chapter = property(_mupdf.fz_location_chapter_get, _mupdf.fz_location_chapter_set)
    page = property(_mupdf.fz_location_page_get, _mupdf.fz_location_page_set)

    def __init__(self):
        _mupdf.fz_location_swiginit(self, _mupdf.new_fz_location())
    __swig_destroy__ = _mupdf.delete_fz_location