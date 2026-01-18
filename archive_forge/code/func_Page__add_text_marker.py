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
def Page__add_text_marker(self, quads, annot_type):
    pdfpage = self._pdf_page()
    rotation = JM_page_rotation(pdfpage)

    def final():
        if rotation != 0:
            mupdf.pdf_dict_put_int(pdfpage.obj(), PDF_NAME('Rotate'), rotation)
    try:
        if rotation != 0:
            mupdf.pdf_dict_put_int(pdfpage.obj(), PDF_NAME('Rotate'), 0)
        annot = mupdf.pdf_create_annot(pdfpage, annot_type)
        for item in quads:
            q = JM_quad_from_py(item)
            mupdf.pdf_add_annot_quad_point(annot, q)
        mupdf.pdf_update_annot(annot)
        JM_add_annot_id(annot, 'A')
        final()
    except Exception:
        if g_exceptions_verbose:
            exception_info()
        final()
        return
    return Annot(annot)