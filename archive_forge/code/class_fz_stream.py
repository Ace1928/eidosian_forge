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
class fz_stream(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    refs = property(_mupdf.fz_stream_refs_get, _mupdf.fz_stream_refs_set)
    error = property(_mupdf.fz_stream_error_get, _mupdf.fz_stream_error_set)
    eof = property(_mupdf.fz_stream_eof_get, _mupdf.fz_stream_eof_set)
    progressive = property(_mupdf.fz_stream_progressive_get, _mupdf.fz_stream_progressive_set)
    pos = property(_mupdf.fz_stream_pos_get, _mupdf.fz_stream_pos_set)
    avail = property(_mupdf.fz_stream_avail_get, _mupdf.fz_stream_avail_set)
    bits = property(_mupdf.fz_stream_bits_get, _mupdf.fz_stream_bits_set)
    rp = property(_mupdf.fz_stream_rp_get, _mupdf.fz_stream_rp_set)
    wp = property(_mupdf.fz_stream_wp_get, _mupdf.fz_stream_wp_set)
    state = property(_mupdf.fz_stream_state_get, _mupdf.fz_stream_state_set)
    next = property(_mupdf.fz_stream_next_get, _mupdf.fz_stream_next_set)
    drop = property(_mupdf.fz_stream_drop_get, _mupdf.fz_stream_drop_set)
    seek = property(_mupdf.fz_stream_seek_get, _mupdf.fz_stream_seek_set)

    def __init__(self):
        _mupdf.fz_stream_swiginit(self, _mupdf.new_fz_stream())
    __swig_destroy__ = _mupdf.delete_fz_stream