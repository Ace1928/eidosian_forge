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
@staticmethod
def _reset_widget(annot):
    this_annot = annot
    this_annot_obj = mupdf.pdf_annot_obj(this_annot)
    pdf = mupdf.pdf_get_bound_document(this_annot_obj)
    mupdf.pdf_field_reset(pdf, this_annot_obj)