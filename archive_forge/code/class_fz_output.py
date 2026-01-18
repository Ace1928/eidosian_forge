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
class fz_output(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    state = property(_mupdf.fz_output_state_get, _mupdf.fz_output_state_set)
    write = property(_mupdf.fz_output_write_get, _mupdf.fz_output_write_set)
    seek = property(_mupdf.fz_output_seek_get, _mupdf.fz_output_seek_set)
    tell = property(_mupdf.fz_output_tell_get, _mupdf.fz_output_tell_set)
    close = property(_mupdf.fz_output_close_get, _mupdf.fz_output_close_set)
    drop = property(_mupdf.fz_output_drop_get, _mupdf.fz_output_drop_set)
    as_stream = property(_mupdf.fz_output_as_stream_get, _mupdf.fz_output_as_stream_set)
    truncate = property(_mupdf.fz_output_truncate_get, _mupdf.fz_output_truncate_set)
    bp = property(_mupdf.fz_output_bp_get, _mupdf.fz_output_bp_set)
    wp = property(_mupdf.fz_output_wp_get, _mupdf.fz_output_wp_set)
    ep = property(_mupdf.fz_output_ep_get, _mupdf.fz_output_ep_set)
    buffered = property(_mupdf.fz_output_buffered_get, _mupdf.fz_output_buffered_set)
    bits = property(_mupdf.fz_output_bits_get, _mupdf.fz_output_bits_set)

    def __init__(self):
        _mupdf.fz_output_swiginit(self, _mupdf.new_fz_output())
    __swig_destroy__ = _mupdf.delete_fz_output