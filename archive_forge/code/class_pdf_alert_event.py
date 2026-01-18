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
class pdf_alert_event(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    doc = property(_mupdf.pdf_alert_event_doc_get, _mupdf.pdf_alert_event_doc_set)
    message = property(_mupdf.pdf_alert_event_message_get, _mupdf.pdf_alert_event_message_set)
    icon_type = property(_mupdf.pdf_alert_event_icon_type_get, _mupdf.pdf_alert_event_icon_type_set)
    button_group_type = property(_mupdf.pdf_alert_event_button_group_type_get, _mupdf.pdf_alert_event_button_group_type_set)
    title = property(_mupdf.pdf_alert_event_title_get, _mupdf.pdf_alert_event_title_set)
    has_check_box = property(_mupdf.pdf_alert_event_has_check_box_get, _mupdf.pdf_alert_event_has_check_box_set)
    check_box_message = property(_mupdf.pdf_alert_event_check_box_message_get, _mupdf.pdf_alert_event_check_box_message_set)
    initially_checked = property(_mupdf.pdf_alert_event_initially_checked_get, _mupdf.pdf_alert_event_initially_checked_set)
    finally_checked = property(_mupdf.pdf_alert_event_finally_checked_get, _mupdf.pdf_alert_event_finally_checked_set)
    button_pressed = property(_mupdf.pdf_alert_event_button_pressed_get, _mupdf.pdf_alert_event_button_pressed_set)

    def __init__(self):
        _mupdf.pdf_alert_event_swiginit(self, _mupdf.new_pdf_alert_event())
    __swig_destroy__ = _mupdf.delete_pdf_alert_event