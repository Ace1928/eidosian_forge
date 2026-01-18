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
def _addWidget(self, field_type, field_name):
    page = self._pdf_page()
    pdf = page.doc()
    annot = JM_create_widget(pdf, page, field_type, field_name)
    if not annot.m_internal:
        raise RuntimeError('cannot create widget')
    JM_add_annot_id(annot, 'W')
    return Annot(annot)