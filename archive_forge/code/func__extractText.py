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
def _extractText(self, format_):
    this_tpage = self.this
    res = mupdf.fz_new_buffer(1024)
    out = mupdf.FzOutput(res)
    if format_ == 1:
        mupdf.fz_print_stext_page_as_html(out, this_tpage, 0)
    elif format_ == 3:
        mupdf.fz_print_stext_page_as_xml(out, this_tpage, 0)
    elif format_ == 4:
        mupdf.fz_print_stext_page_as_xhtml(out, this_tpage, 0)
    else:
        JM_print_stext_page_as_text(res, this_tpage)
    out.fz_close_output()
    text = JM_EscapeStrFromBuffer(res)
    return text