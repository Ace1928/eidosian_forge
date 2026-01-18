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
def Tools__parse_da(annot):
    this_annot = annot.this
    assert isinstance(this_annot, mupdf.PdfAnnot)
    this_annot_obj = mupdf.pdf_annot_obj(this_annot)
    pdf = mupdf.pdf_get_bound_document(this_annot_obj)
    try:
        da = mupdf.pdf_dict_get_inheritable(this_annot_obj, PDF_NAME('DA'))
        if not da.m_internal:
            trailer = mupdf.pdf_trailer(pdf)
            da = mupdf.pdf_dict_getl(trailer, PDF_NAME('Root'), PDF_NAME('AcroForm'), PDF_NAME('DA'))
        da_str = mupdf.pdf_to_text_string(da)
    except Exception:
        if g_exceptions_verbose:
            exception_info()
        return
    return da_str