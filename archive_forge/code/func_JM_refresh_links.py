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
def JM_refresh_links(page):
    """
    refreshes the link and annotation tables of a page
    """
    if not page:
        return
    obj = mupdf.pdf_dict_get(page.obj(), PDF_NAME('Annots'))
    if obj.m_internal:
        pdf = page.doc()
        number = mupdf.pdf_lookup_page_number(pdf, page.obj())
        page_mediabox = mupdf.FzRect()
        page_ctm = mupdf.FzMatrix()
        mupdf.pdf_page_transform(page, page_mediabox, page_ctm)
        link = mupdf.pdf_load_link_annots(pdf, page, obj, number, page_ctm)
        page.m_internal.links = mupdf.ll_fz_keep_link(link.m_internal)