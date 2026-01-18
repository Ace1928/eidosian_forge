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
def JM_mediabox(page_obj):
    """
    return a PDF page's MediaBox
    """
    page_mediabox = mupdf.FzRect(mupdf.FzRect.Fixed_UNIT)
    mediabox = mupdf.pdf_to_rect(mupdf.pdf_dict_get_inheritable(page_obj, PDF_NAME('MediaBox')))
    if mupdf.fz_is_empty_rect(mediabox) or mupdf.fz_is_infinite_rect(mediabox):
        mediabox.x0 = 0
        mediabox.y0 = 0
        mediabox.x1 = 612
        mediabox.y1 = 792
    page_mediabox = mupdf.FzRect(mupdf.fz_min(mediabox.x0, mediabox.x1), mupdf.fz_min(mediabox.y0, mediabox.y1), mupdf.fz_max(mediabox.x0, mediabox.x1), mupdf.fz_max(mediabox.y0, mediabox.y1))
    if page_mediabox.x1 - page_mediabox.x0 < 1 or page_mediabox.y1 - page_mediabox.y0 < 1:
        page_mediabox = mupdf.FzRect(mupdf.FzRect.Fixed_UNIT)
    return page_mediabox