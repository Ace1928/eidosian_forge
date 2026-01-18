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
def JM_get_ocg_arrays_imp(arr):
    """
    Get OCG arrays from OC configuration
    Returns dict {"basestate":name, "on":list, "off":list, "rbg":list, "locked":list}
    """
    list_ = list()
    if mupdf.pdf_is_array(arr):
        n = mupdf.pdf_array_len(arr)
        for i in range(n):
            obj = mupdf.pdf_array_get(arr, i)
            item = mupdf.pdf_to_num(obj)
            if item not in list_:
                list_.append(item)
    return list_