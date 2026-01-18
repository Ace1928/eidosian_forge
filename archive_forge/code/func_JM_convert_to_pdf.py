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
def JM_convert_to_pdf(doc, fp, tp, rotate):
    """
    Convert any MuPDF document to a PDF
    Returns bytes object containing the PDF, created via 'write' function.
    """
    pdfout = mupdf.PdfDocument()
    incr = 1
    s = fp
    e = tp
    if fp > tp:
        incr = -1
        s = tp
        e = fp
    rot = JM_norm_rotation(rotate)
    i = fp
    while 1:
        if not _INRANGE(i, s, e):
            break
        page = mupdf.fz_load_page(doc, i)
        mediabox = mupdf.fz_bound_page(page)
        dev, resources, contents = mupdf.pdf_page_write(pdfout, mediabox)
        mupdf.fz_run_page(page, dev, mupdf.FzMatrix(), mupdf.FzCookie())
        mupdf.fz_close_device(dev)
        dev = None
        page_obj = mupdf.pdf_add_page(pdfout, mediabox, rot, resources, contents)
        mupdf.pdf_insert_page(pdfout, -1, page_obj)
        i += incr
    opts = mupdf.PdfWriteOptions()
    opts.do_garbage = 4
    opts.do_compress = 1
    opts.do_compress_images = 1
    opts.do_compress_fonts = 1
    opts.do_sanitize = 1
    opts.do_incremental = 0
    opts.do_ascii = 0
    opts.do_decompress = 0
    opts.do_linear = 0
    opts.do_clean = 1
    opts.do_pretty = 0
    res = mupdf.fz_new_buffer(8192)
    out = mupdf.FzOutput(res)
    mupdf.pdf_write_document(pdfout, out, opts)
    out.fz_close_output()
    c = mupdf.fz_buffer_extract_copy(res)
    assert isinstance(c, bytes)
    return c