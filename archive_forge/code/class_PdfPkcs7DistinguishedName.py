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
class PdfPkcs7DistinguishedName(object):
    """ Wrapper class for struct `pdf_pkcs7_distinguished_name`. Not copyable or assignable."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_signature_drop_distinguished_name(self):
        """ Class-aware wrapper for `::pdf_signature_drop_distinguished_name()`."""
        return _mupdf.PdfPkcs7DistinguishedName_pdf_signature_drop_distinguished_name(self)

    def pdf_signature_format_distinguished_name(self):
        """ Class-aware wrapper for `::pdf_signature_format_distinguished_name()`."""
        return _mupdf.PdfPkcs7DistinguishedName_pdf_signature_format_distinguished_name(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        Default constructor, sets `m_internal` to null.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::pdf_pkcs7_distinguished_name`.
        """
        _mupdf.PdfPkcs7DistinguishedName_swiginit(self, _mupdf.new_PdfPkcs7DistinguishedName(*args))
    __swig_destroy__ = _mupdf.delete_PdfPkcs7DistinguishedName

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfPkcs7DistinguishedName_m_internal_value(self)
    m_internal = property(_mupdf.PdfPkcs7DistinguishedName_m_internal_get, _mupdf.PdfPkcs7DistinguishedName_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfPkcs7DistinguishedName_s_num_instances_get, _mupdf.PdfPkcs7DistinguishedName_s_num_instances_set)