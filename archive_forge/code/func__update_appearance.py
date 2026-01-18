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
def _update_appearance(self, opacity=-1, blend_mode=None, fill_color=None, rotate=-1):
    annot = self.this
    assert annot.m_internal
    annot_obj = mupdf.pdf_annot_obj(annot)
    page = mupdf.pdf_annot_page(annot)
    pdf = page.doc()
    type_ = mupdf.pdf_annot_type(annot)
    nfcol, fcol = JM_color_FromSequence(fill_color)
    try:
        if nfcol == 0 or type_ not in (mupdf.PDF_ANNOT_SQUARE, mupdf.PDF_ANNOT_CIRCLE, mupdf.PDF_ANNOT_LINE, mupdf.PDF_ANNOT_POLY_LINE, mupdf.PDF_ANNOT_POLYGON):
            mupdf.pdf_dict_del(annot_obj, PDF_NAME('IC'))
        elif nfcol > 0:
            mupdf.pdf_set_annot_interior_color(annot, fcol[:nfcol])
        insert_rot = 1 if rotate >= 0 else 0
        if type_ not in (mupdf.PDF_ANNOT_CARET, mupdf.PDF_ANNOT_CIRCLE, mupdf.PDF_ANNOT_FREE_TEXT, mupdf.PDF_ANNOT_FILE_ATTACHMENT, mupdf.PDF_ANNOT_INK, mupdf.PDF_ANNOT_LINE, mupdf.PDF_ANNOT_POLY_LINE, mupdf.PDF_ANNOT_POLYGON, mupdf.PDF_ANNOT_SQUARE, mupdf.PDF_ANNOT_STAMP, mupdf.PDF_ANNOT_TEXT):
            insert_rot = 0
        if insert_rot:
            mupdf.pdf_dict_put_int(annot_obj, PDF_NAME('Rotate'), rotate)
        mupdf.pdf_dirty_annot(annot)
        mupdf.pdf_update_annot(annot)
        pdf.resynth_required = 0
        if type_ == mupdf.PDF_ANNOT_FREE_TEXT:
            if nfcol > 0:
                mupdf.pdf_set_annot_color(annot, fcol[:nfcol])
        elif nfcol > 0:
            col = mupdf.pdf_new_array(page.doc(), nfcol)
            for i in range(nfcol):
                mupdf.pdf_array_push_real(col, fcol[i])
            mupdf.pdf_dict_put(annot_obj, PDF_NAME('IC'), col)
    except Exception as e:
        if g_exceptions_verbose:
            exception_info()
        message(f'cannot update annot: {e}', file=sys.stderr)
        raise
    if (opacity < 0 or opacity >= 1) and (not blend_mode):
        return True
    try:
        ap = mupdf.pdf_dict_getl(mupdf.pdf_annot_obj(annot), PDF_NAME('AP'), PDF_NAME('N'))
        if not ap.m_internal:
            raise RuntimeError(MSG_BAD_APN)
        resources = mupdf.pdf_dict_get(ap, PDF_NAME('Resources'))
        if not resources.m_internal:
            resources = mupdf.pdf_dict_put_dict(ap, PDF_NAME('Resources'), 2)
        alp0 = mupdf.pdf_new_dict(page.doc(), 3)
        if opacity >= 0 and opacity < 1:
            mupdf.pdf_dict_put_real(alp0, PDF_NAME('CA'), opacity)
            mupdf.pdf_dict_put_real(alp0, PDF_NAME('ca'), opacity)
            mupdf.pdf_dict_put_real(annot_obj, PDF_NAME('CA'), opacity)
        if blend_mode:
            mupdf.pdf_dict_put_name(alp0, PDF_NAME('BM'), blend_mode)
            mupdf.pdf_dict_put_name(annot_obj, PDF_NAME('BM'), blend_mode)
        extg = mupdf.pdf_dict_get(resources, PDF_NAME('ExtGState'))
        if not extg.m_internal:
            extg = mupdf.pdf_dict_put_dict(resources, PDF_NAME('ExtGState'), 2)
        mupdf.pdf_dict_put(extg, PDF_NAME('H'), alp0)
    except Exception as e:
        if g_exceptions_verbose:
            exception_info()
        message(f'cannot set opacity or blend mode\n: {e}', file=sys.stderr)
        raise
    return True