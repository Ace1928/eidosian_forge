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
class pdf_pattern(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    storable = property(_mupdf.pdf_pattern_storable_get, _mupdf.pdf_pattern_storable_set)
    ismask = property(_mupdf.pdf_pattern_ismask_get, _mupdf.pdf_pattern_ismask_set)
    xstep = property(_mupdf.pdf_pattern_xstep_get, _mupdf.pdf_pattern_xstep_set)
    ystep = property(_mupdf.pdf_pattern_ystep_get, _mupdf.pdf_pattern_ystep_set)
    matrix = property(_mupdf.pdf_pattern_matrix_get, _mupdf.pdf_pattern_matrix_set)
    bbox = property(_mupdf.pdf_pattern_bbox_get, _mupdf.pdf_pattern_bbox_set)
    document = property(_mupdf.pdf_pattern_document_get, _mupdf.pdf_pattern_document_set)
    resources = property(_mupdf.pdf_pattern_resources_get, _mupdf.pdf_pattern_resources_set)
    contents = property(_mupdf.pdf_pattern_contents_get, _mupdf.pdf_pattern_contents_set)
    id = property(_mupdf.pdf_pattern_id_get, _mupdf.pdf_pattern_id_set)

    def __init__(self):
        _mupdf.pdf_pattern_swiginit(self, _mupdf.new_pdf_pattern())
    __swig_destroy__ = _mupdf.delete_pdf_pattern