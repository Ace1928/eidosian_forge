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
class pdf_pkcs7_verifier(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    drop = property(_mupdf.pdf_pkcs7_verifier_drop_get, _mupdf.pdf_pkcs7_verifier_drop_set)
    check_certificate = property(_mupdf.pdf_pkcs7_verifier_check_certificate_get, _mupdf.pdf_pkcs7_verifier_check_certificate_set)
    check_digest = property(_mupdf.pdf_pkcs7_verifier_check_digest_get, _mupdf.pdf_pkcs7_verifier_check_digest_set)
    get_signatory = property(_mupdf.pdf_pkcs7_verifier_get_signatory_get, _mupdf.pdf_pkcs7_verifier_get_signatory_set)

    def __init__(self):
        _mupdf.pdf_pkcs7_verifier_swiginit(self, _mupdf.new_pdf_pkcs7_verifier())
    __swig_destroy__ = _mupdf.delete_pdf_pkcs7_verifier