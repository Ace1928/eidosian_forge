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
def JM_set_ocg_arrays(conf, basestate, on, off, rbgroups, locked):
    if basestate:
        mupdf.pdf_dict_put_name(conf, PDF_NAME('BaseState'), basestate)
    if on is not None:
        mupdf.pdf_dict_del(conf, PDF_NAME('ON'))
        if on:
            arr = mupdf.pdf_dict_put_array(conf, PDF_NAME('ON'), 1)
            JM_set_ocg_arrays_imp(arr, on)
    if off is not None:
        mupdf.pdf_dict_del(conf, PDF_NAME('OFF'))
        if off:
            arr = mupdf.pdf_dict_put_array(conf, PDF_NAME('OFF'), 1)
            JM_set_ocg_arrays_imp(arr, off)
    if locked is not None:
        mupdf.pdf_dict_del(conf, PDF_NAME('Locked'))
        if locked:
            arr = mupdf.pdf_dict_put_array(conf, PDF_NAME('Locked'), 1)
            JM_set_ocg_arrays_imp(arr, locked)
    if rbgroups is not None:
        mupdf.pdf_dict_del(conf, PDF_NAME('RBGroups'))
        if rbgroups:
            arr = mupdf.pdf_dict_put_array(conf, PDF_NAME('RBGroups'), 1)
            n = len(rbgroups)
            for i in range(n):
                item0 = rbgroups[i]
                obj = mupdf.pdf_array_push_array(arr, 1)
                JM_set_ocg_arrays_imp(obj, item0)