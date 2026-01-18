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
def _embeddedFileGet(self, idx):
    pdf = _as_pdf_document(self)
    names = mupdf.pdf_dict_getl(mupdf.pdf_trailer(pdf), PDF_NAME('Root'), PDF_NAME('Names'), PDF_NAME('EmbeddedFiles'), PDF_NAME('Names'))
    entry = mupdf.pdf_array_get(names, 2 * idx + 1)
    filespec = mupdf.pdf_dict_getl(entry, PDF_NAME('EF'), PDF_NAME('F'))
    buf = mupdf.pdf_load_stream(filespec)
    cont = JM_BinFromBuffer(buf)
    return cont