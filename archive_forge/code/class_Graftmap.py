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
class Graftmap:

    def __del__(self):
        if not type(self) is Graftmap:
            return
        self.thisown = False

    def __init__(self, doc):
        dst = _as_pdf_document(doc)
        ASSERT_PDF(dst)
        map_ = mupdf.pdf_new_graft_map(dst)
        self.this = map_
        self.thisown = True