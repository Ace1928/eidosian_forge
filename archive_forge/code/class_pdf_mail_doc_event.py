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
class pdf_mail_doc_event(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    ask_user = property(_mupdf.pdf_mail_doc_event_ask_user_get, _mupdf.pdf_mail_doc_event_ask_user_set)
    to = property(_mupdf.pdf_mail_doc_event_to_get, _mupdf.pdf_mail_doc_event_to_set)
    cc = property(_mupdf.pdf_mail_doc_event_cc_get, _mupdf.pdf_mail_doc_event_cc_set)
    bcc = property(_mupdf.pdf_mail_doc_event_bcc_get, _mupdf.pdf_mail_doc_event_bcc_set)
    subject = property(_mupdf.pdf_mail_doc_event_subject_get, _mupdf.pdf_mail_doc_event_subject_set)
    message = property(_mupdf.pdf_mail_doc_event_message_get, _mupdf.pdf_mail_doc_event_message_set)

    def __init__(self):
        _mupdf.pdf_mail_doc_event_swiginit(self, _mupdf.new_pdf_mail_doc_event())
    __swig_destroy__ = _mupdf.delete_pdf_mail_doc_event