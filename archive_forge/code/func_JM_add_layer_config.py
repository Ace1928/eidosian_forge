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
def JM_add_layer_config(pdf, name, creator, ON):
    """
    Add OC configuration to the PDF catalog
    """
    ocp = JM_ensure_ocproperties(pdf)
    configs = mupdf.pdf_dict_get(ocp, PDF_NAME('Configs'))
    if not mupdf.pdf_is_array(configs):
        configs = mupdf.pdf_dict_put_array(ocp, PDF_NAME('Configs'), 1)
    D = mupdf.pdf_new_dict(pdf, 5)
    mupdf.pdf_dict_put_text_string(D, PDF_NAME('Name'), name)
    if creator is not None:
        mupdf.pdf_dict_put_text_string(D, PDF_NAME('Creator'), creator)
    mupdf.pdf_dict_put(D, PDF_NAME('BaseState'), PDF_NAME('OFF'))
    onarray = mupdf.pdf_dict_put_array(D, PDF_NAME('ON'), 5)
    if not ON:
        pass
    else:
        ocgs = mupdf.pdf_dict_get(ocp, PDF_NAME('OCGs'))
        n = len(ON)
        for i in range(n):
            xref = 0
            e, xref = JM_INT_ITEM(ON, i)
            if e == 1:
                continue
            ind = mupdf.pdf_new_indirect(pdf, xref, 0)
            if mupdf.pdf_array_contains(ocgs, ind):
                mupdf.pdf_array_push(onarray, ind)
    mupdf.pdf_array_push(configs, D)