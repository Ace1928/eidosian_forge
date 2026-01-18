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
def _as_pdf_page(page):
    """
    Returns page as a mupdf.PdfPage, downcasting as required. If we fail (i.e.
    page is a mupdf.FzPage(), <ret>.m_internal will be None.
    """
    if isinstance(page, Page):
        page = page.this
    if isinstance(page, mupdf.PdfPage):
        return page
    elif isinstance(page, mupdf.FzPage):
        return mupdf.pdf_page_from_fz_page(page)
    elif page is None:
        assert 0, f'page is None'
    else:
        assert 0, f'Unrecognised type(page)={type(page)!r}'