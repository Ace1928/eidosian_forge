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
def JM_set_ocg_arrays_imp(arr, list_):
    """
    Set OCG arrays from dict of Python lists
    Works with dict like {"basestate":name, "on":list, "off":list, "rbg":list}
    """
    pdf = mupdf.pdf_get_bound_document(arr)
    for xref in list_:
        obj = mupdf.pdf_new_indirect(pdf, xref, 0)
        mupdf.pdf_array_push(arr, obj)