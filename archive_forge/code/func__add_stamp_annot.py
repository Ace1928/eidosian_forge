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
def _add_stamp_annot(self, rect, stamp=0):
    page = self._pdf_page()
    stamp_id = [PDF_NAME('Approved'), PDF_NAME('AsIs'), PDF_NAME('Confidential'), PDF_NAME('Departmental'), PDF_NAME('Experimental'), PDF_NAME('Expired'), PDF_NAME('Final'), PDF_NAME('ForComment'), PDF_NAME('ForPublicRelease'), PDF_NAME('NotApproved'), PDF_NAME('NotForPublicRelease'), PDF_NAME('Sold'), PDF_NAME('TopSecret'), PDF_NAME('Draft')]
    n = len(stamp_id)
    name = stamp_id[0]
    ASSERT_PDF(page)
    r = JM_rect_from_py(rect)
    if mupdf.fz_is_infinite_rect(r) or mupdf.fz_is_empty_rect(r):
        raise ValueError(MSG_BAD_RECT)
    if _INRANGE(stamp, 0, n - 1):
        name = stamp_id[stamp]
    annot = mupdf.pdf_create_annot(page, mupdf.PDF_ANNOT_STAMP)
    mupdf.pdf_set_annot_rect(annot, r)
    try:
        n = PDF_NAME('Name')
        mupdf.pdf_dict_put(mupdf.pdf_annot_obj(annot), PDF_NAME('Name'), name)
    except Exception:
        if g_exceptions_verbose:
            exception_info()
        raise
    mupdf.pdf_set_annot_contents(annot, mupdf.pdf_dict_get_name(mupdf.pdf_annot_obj(annot), PDF_NAME('Name')))
    mupdf.pdf_update_annot(annot)
    JM_add_annot_id(annot, 'A')
    return Annot(annot)