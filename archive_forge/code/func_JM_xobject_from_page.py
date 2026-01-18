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
def JM_xobject_from_page(pdfout, fsrcpage, xref, gmap):
    """
    Make an XObject from a PDF page
    For a positive xref assume that its object can be used instead
    """
    assert isinstance(gmap, mupdf.PdfGraftMap), f'type(gmap)={type(gmap)!r}'
    if xref > 0:
        xobj1 = mupdf.pdf_new_indirect(pdfout, xref, 0)
    else:
        srcpage = mupdf.pdf_page_from_fz_page(fsrcpage.this)
        spageref = srcpage.obj()
        mediabox = mupdf.pdf_to_rect(mupdf.pdf_dict_get_inheritable(spageref, PDF_NAME('MediaBox')))
        o = mupdf.pdf_dict_get_inheritable(spageref, PDF_NAME('Resources'))
        if gmap.m_internal:
            resources = mupdf.pdf_graft_mapped_object(gmap, o)
        else:
            resources = mupdf.pdf_graft_object(pdfout, o)
        res = JM_read_contents(spageref)
        xobj1 = mupdf.pdf_new_xobject(pdfout, mediabox, mupdf.FzMatrix(), mupdf.PdfObj(0), res)
        JM_update_stream(pdfout, xobj1, res, 1)
        mupdf.pdf_dict_put(xobj1, PDF_NAME('Resources'), resources)
    return xobj1