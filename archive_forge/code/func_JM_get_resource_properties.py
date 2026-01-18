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
def JM_get_resource_properties(ref):
    """
    Return the items of Resources/Properties (used for Marked Content)
    Argument may be e.g. a page object or a Form XObject
    """
    properties = mupdf.pdf_dict_getl(ref, PDF_NAME('Resources'), PDF_NAME('Properties'))
    if not properties.m_internal:
        return ()
    else:
        n = mupdf.pdf_dict_len(properties)
        if n < 1:
            return ()
        rc = []
        for i in range(n):
            key = mupdf.pdf_dict_get_key(properties, i)
            val = mupdf.pdf_dict_get_val(properties, i)
            c = mupdf.pdf_to_name(key)
            xref = mupdf.pdf_to_num(val)
            rc.append((c, xref))
    return rc