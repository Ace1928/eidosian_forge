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
def JM_ensure_identity(pdf):
    """
    Store ID in PDF trailer
    """
    id_ = mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('ID'))
    if not id_.m_internal:
        rnd0 = mupdf.fz_memrnd2(16)
        rnd = ''
        for i in rnd0:
            rnd += chr(i)
        id_ = mupdf.pdf_dict_put_array(mupdf.pdf_trailer(pdf), PDF_NAME('ID'), 2)
        mupdf.pdf_array_push(id_, mupdf.pdf_new_string(rnd, len(rnd)))
        mupdf.pdf_array_push(id_, mupdf.pdf_new_string(rnd, len(rnd)))