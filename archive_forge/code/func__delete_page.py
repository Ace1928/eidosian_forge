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
def _delete_page(self, pno):
    pdf = _as_pdf_document(self)
    mupdf.pdf_delete_page(pdf, pno)
    if pdf.m_internal.rev_page_map:
        mupdf.ll_pdf_drop_page_tree(pdf.m_internal)