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
def JM_ensure_ocproperties(pdf):
    """
    Ensure OCProperties, return /OCProperties key
    """
    ocp = mupdf.pdf_dict_get(mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('Root')), PDF_NAME('OCProperties'))
    if ocp.m_internal:
        return ocp
    root = mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('Root'))
    ocp = mupdf.pdf_dict_put_dict(root, PDF_NAME('OCProperties'), 2)
    mupdf.pdf_dict_put_array(ocp, PDF_NAME('OCGs'), 0)
    D = mupdf.pdf_dict_put_dict(ocp, PDF_NAME('D'), 5)
    mupdf.pdf_dict_put_array(D, PDF_NAME('ON'), 0)
    mupdf.pdf_dict_put_array(D, PDF_NAME('OFF'), 0)
    mupdf.pdf_dict_put_array(D, PDF_NAME('Order'), 0)
    mupdf.pdf_dict_put_array(D, PDF_NAME('RBGroups'), 0)
    return ocp