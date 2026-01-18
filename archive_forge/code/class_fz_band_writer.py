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
class fz_band_writer(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    drop = property(_mupdf.fz_band_writer_drop_get, _mupdf.fz_band_writer_drop_set)
    close = property(_mupdf.fz_band_writer_close_get, _mupdf.fz_band_writer_close_set)
    header = property(_mupdf.fz_band_writer_header_get, _mupdf.fz_band_writer_header_set)
    band = property(_mupdf.fz_band_writer_band_get, _mupdf.fz_band_writer_band_set)
    trailer = property(_mupdf.fz_band_writer_trailer_get, _mupdf.fz_band_writer_trailer_set)
    out = property(_mupdf.fz_band_writer_out_get, _mupdf.fz_band_writer_out_set)
    w = property(_mupdf.fz_band_writer_w_get, _mupdf.fz_band_writer_w_set)
    h = property(_mupdf.fz_band_writer_h_get, _mupdf.fz_band_writer_h_set)
    n = property(_mupdf.fz_band_writer_n_get, _mupdf.fz_band_writer_n_set)
    s = property(_mupdf.fz_band_writer_s_get, _mupdf.fz_band_writer_s_set)
    alpha = property(_mupdf.fz_band_writer_alpha_get, _mupdf.fz_band_writer_alpha_set)
    xres = property(_mupdf.fz_band_writer_xres_get, _mupdf.fz_band_writer_xres_set)
    yres = property(_mupdf.fz_band_writer_yres_get, _mupdf.fz_band_writer_yres_set)
    pagenum = property(_mupdf.fz_band_writer_pagenum_get, _mupdf.fz_band_writer_pagenum_set)
    line = property(_mupdf.fz_band_writer_line_get, _mupdf.fz_band_writer_line_set)
    seps = property(_mupdf.fz_band_writer_seps_get, _mupdf.fz_band_writer_seps_set)

    def __init__(self):
        _mupdf.fz_band_writer_swiginit(self, _mupdf.new_fz_band_writer())
    __swig_destroy__ = _mupdf.delete_fz_band_writer