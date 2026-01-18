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
def JM_set_object_value(obj, key, value):
    """
    Set a PDF dict key to some value
    """
    eyecatcher = 'fitz: replace me!'
    pdf = mupdf.pdf_get_bound_document(obj)
    list_ = key.split('/')
    len_ = len(list_)
    i = len_ - 1
    skey = list_[i]
    del list_[i]
    len_ = len(list_)
    testkey = mupdf.pdf_dict_getp(obj, key)
    if not testkey.m_internal:
        while len_ > 0:
            t = '/'.join(list_)
            if mupdf.pdf_is_indirect(mupdf.pdf_dict_getp(obj, JM_StrAsChar(t))):
                raise Exception("path to '%s' has indirects", JM_StrAsChar(skey))
            del list_[len_ - 1]
            len_ = len(list_)
    mupdf.pdf_dict_putp(obj, key, mupdf.pdf_new_text_string(eyecatcher))
    testkey = mupdf.pdf_dict_getp(obj, key)
    if not mupdf.pdf_is_string(testkey):
        raise Exception("cannot insert value for '%s'", key)
    temp = mupdf.pdf_to_text_string(testkey)
    if temp != eyecatcher:
        raise Exception("cannot insert value for '%s'", key)
    res = JM_object_to_buffer(obj, 1, 0)
    objstr = JM_EscapeStrFromBuffer(res)
    nullval = '/%s(%s)' % (skey, eyecatcher)
    newval = '/%s %s' % (skey, value)
    newstr = objstr.replace(nullval, newval, 1)
    new_obj = JM_pdf_obj_from_str(pdf, newstr)
    return new_obj