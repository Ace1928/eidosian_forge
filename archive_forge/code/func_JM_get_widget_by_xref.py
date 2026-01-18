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
def JM_get_widget_by_xref(page, xref):
    """
    retrieve widget by its xref
    """
    found = False
    annot = mupdf.pdf_first_widget(page)
    while annot.m_internal:
        annot_obj = mupdf.pdf_annot_obj(annot)
        if xref == mupdf.pdf_to_num(annot_obj):
            found = True
            break
        annot = mupdf.pdf_next_widget(annot)
    if not found:
        raise Exception(f'xref {xref} is not a widget of this page')
    return Annot(annot)