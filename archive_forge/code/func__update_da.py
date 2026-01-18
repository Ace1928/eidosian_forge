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
def _update_da(annot, da_str):
    if g_use_extra:
        extra.Tools_update_da(annot.this, da_str)
    else:
        try:
            this_annot = annot.this
            assert isinstance(this_annot, mupdf.PdfAnnot)
            mupdf.pdf_dict_put_text_string(mupdf.pdf_annot_obj(this_annot), PDF_NAME('DA'), da_str)
            mupdf.pdf_dict_del(mupdf.pdf_annot_obj(this_annot), PDF_NAME('DS'))
            mupdf.pdf_dict_del(mupdf.pdf_annot_obj(this_annot), PDF_NAME('RC'))
        except Exception:
            if g_exceptions_verbose:
                exception_info()
            return
        return