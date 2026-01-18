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
def JM_outline_xrefs(obj, xrefs):
    """
    Return list of outline xref numbers. Recursive function. Arguments:
    'obj' first OL item
    'xrefs' empty Python list
    """
    if not obj.m_internal:
        return xrefs
    thisobj = obj
    while thisobj.m_internal:
        newxref = mupdf.pdf_to_num(thisobj)
        if newxref in xrefs or mupdf.pdf_dict_get(thisobj, PDF_NAME('Type')).m_internal:
            break
        xrefs.append(newxref)
        first = mupdf.pdf_dict_get(thisobj, PDF_NAME('First'))
        if mupdf.pdf_is_dict(first):
            xrefs = JM_outline_xrefs(first, xrefs)
        thisobj = mupdf.pdf_dict_get(thisobj, PDF_NAME('Next'))
        parent = mupdf.pdf_dict_get(thisobj, PDF_NAME('Parent'))
        if not mupdf.pdf_is_dict(thisobj):
            thisobj = parent
    return xrefs