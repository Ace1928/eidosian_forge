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
class pdf_keystroke_event(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    value = property(_mupdf.pdf_keystroke_event_value_get, _mupdf.pdf_keystroke_event_value_set)
    change = property(_mupdf.pdf_keystroke_event_change_get, _mupdf.pdf_keystroke_event_change_set)
    selStart = property(_mupdf.pdf_keystroke_event_selStart_get, _mupdf.pdf_keystroke_event_selStart_set)
    selEnd = property(_mupdf.pdf_keystroke_event_selEnd_get, _mupdf.pdf_keystroke_event_selEnd_set)
    willCommit = property(_mupdf.pdf_keystroke_event_willCommit_get, _mupdf.pdf_keystroke_event_willCommit_set)
    newChange = property(_mupdf.pdf_keystroke_event_newChange_get, _mupdf.pdf_keystroke_event_newChange_set)
    newValue = property(_mupdf.pdf_keystroke_event_newValue_get, _mupdf.pdf_keystroke_event_newValue_set)

    def __init__(self):
        _mupdf.pdf_keystroke_event_swiginit(self, _mupdf.new_pdf_keystroke_event())
    __swig_destroy__ = _mupdf.delete_pdf_keystroke_event