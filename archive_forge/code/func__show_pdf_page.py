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
def _show_pdf_page(self, fz_srcpage, overlay=1, matrix=None, xref=0, oc=0, clip=None, graftmap=None, _imgname=None):
    cropbox = JM_rect_from_py(clip)
    mat = JM_matrix_from_py(matrix)
    rc_xref = xref
    tpage = mupdf.pdf_page_from_fz_page(self.this)
    tpageref = tpage.obj()
    pdfout = tpage.doc()
    ENSURE_OPERATION(pdfout)
    xobj1 = JM_xobject_from_page(pdfout, fz_srcpage, xref, graftmap.this)
    if not rc_xref:
        rc_xref = mupdf.pdf_to_num(xobj1)
    subres1 = mupdf.pdf_new_dict(pdfout, 5)
    mupdf.pdf_dict_puts(subres1, 'fullpage', xobj1)
    subres = mupdf.pdf_new_dict(pdfout, 5)
    mupdf.pdf_dict_put(subres, PDF_NAME('XObject'), subres1)
    res = mupdf.fz_new_buffer(20)
    mupdf.fz_append_string(res, '/fullpage Do')
    xobj2 = mupdf.pdf_new_xobject(pdfout, cropbox, mat, subres, res)
    if oc > 0:
        JM_add_oc_object(pdfout, mupdf.pdf_resolve_indirect(xobj2), oc)
    resources = mupdf.pdf_dict_get_inheritable(tpageref, PDF_NAME('Resources'))
    subres = mupdf.pdf_dict_get(resources, PDF_NAME('XObject'))
    if not subres.m_internal:
        subres = mupdf.pdf_dict_put_dict(resources, PDF_NAME('XObject'), 5)
    mupdf.pdf_dict_puts(subres, _imgname, xobj2)
    nres = mupdf.fz_new_buffer(50)
    mupdf.fz_append_string(nres, ' q /')
    mupdf.fz_append_string(nres, _imgname)
    mupdf.fz_append_string(nres, ' Do Q ')
    JM_insert_contents(pdfout, tpageref, nres, overlay)
    return rc_xref