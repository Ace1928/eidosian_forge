import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def JM_create_widget(doc, page, type, fieldname):
    old_sigflags = mupdf.pdf_to_int(mupdf.pdf_dict_getp(mupdf.pdf_trailer(doc), 'Root/AcroForm/SigFlags'))
    annot = mupdf.pdf_create_annot_raw(page, mupdf.PDF_ANNOT_WIDGET)
    annot_obj = mupdf.pdf_annot_obj(annot)
    try:
        JM_set_field_type(doc, annot_obj, type)
        mupdf.pdf_dict_put_text_string(annot_obj, PDF_NAME('T'), fieldname)
        if type == mupdf.PDF_WIDGET_TYPE_SIGNATURE:
            sigflags = old_sigflags | (SigFlag_SignaturesExist | SigFlag_AppendOnly)
            mupdf.pdf_dict_putl(mupdf.pdf_trailer(doc), mupdf.pdf_new_nt(sigflags), PDF_NAME('Root'), PDF_NAME('AcroForm'), PDF_NAME('SigFlags'))
        form = mupdf.pdf_dict_getp(mupdf.pdf_trailer(doc), 'Root/AcroForm/Fields')
        if not form.m_internal:
            form = mupdf.pdf_new_array(doc, 1)
            mupdf.pdf_dict_putl(mupdf.pdf_trailer(doc), form, PDF_NAME('Root'), PDF_NAME('AcroForm'), PDF_NAME('Fields'))
        mupdf.pdf_array_push(form, annot_obj)
    except Exception:
        if g_exceptions_verbose:
            exception_info()
        mupdf.pdf_delete_annot(page, annot)
        if type == mupdf.PDF_WIDGET_TYPE_SIGNATURE:
            mupdf.pdf_dict_putl(mupdf.pdf_trailer(doc), mupdf.pdf_new_int(old_sigflags), PDF_NAME('Root'), PDF_NAME('AcroForm'), PDF_NAME('SigFlags'))
        raise
    return annot