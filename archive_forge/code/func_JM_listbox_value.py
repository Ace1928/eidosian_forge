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
def JM_listbox_value(annot):
    """
    ListBox retrieve value
    """
    annot_obj = mupdf.pdf_annot_obj(annot)
    optarr = mupdf.pdf_dict_get(annot_obj, PDF_NAME('V'))
    if mupdf.pdf_is_string(optarr):
        return mupdf.pdf_to_text_string(optarr)
    n = mupdf.pdf_array_len(optarr)
    liste = []
    for i in range(n):
        elem = mupdf.pdf_array_get(optarr, i)
        if mupdf.pdf_is_array(elem):
            elem = mupdf.pdf_array_get(elem, 1)
        liste.append(JM_UnicodeFromStr(mupdf.pdf_to_text_string(elem)))
    return liste