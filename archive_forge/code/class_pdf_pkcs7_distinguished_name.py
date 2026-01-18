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
class pdf_pkcs7_distinguished_name(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    cn = property(_mupdf.pdf_pkcs7_distinguished_name_cn_get, _mupdf.pdf_pkcs7_distinguished_name_cn_set)
    o = property(_mupdf.pdf_pkcs7_distinguished_name_o_get, _mupdf.pdf_pkcs7_distinguished_name_o_set)
    ou = property(_mupdf.pdf_pkcs7_distinguished_name_ou_get, _mupdf.pdf_pkcs7_distinguished_name_ou_set)
    email = property(_mupdf.pdf_pkcs7_distinguished_name_email_get, _mupdf.pdf_pkcs7_distinguished_name_email_set)
    c = property(_mupdf.pdf_pkcs7_distinguished_name_c_get, _mupdf.pdf_pkcs7_distinguished_name_c_set)

    def __init__(self):
        _mupdf.pdf_pkcs7_distinguished_name_swiginit(self, _mupdf.new_pdf_pkcs7_distinguished_name())
    __swig_destroy__ = _mupdf.delete_pdf_pkcs7_distinguished_name