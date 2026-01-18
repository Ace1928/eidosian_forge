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
def _add_multiline(self, points, annot_type):
    page = self._pdf_page()
    if len(points) < 2:
        raise ValueError(MSG_BAD_ARG_POINTS)
    annot = mupdf.pdf_create_annot(page, annot_type)
    for p in points:
        if PySequence_Size(p) != 2:
            raise ValueError(MSG_BAD_ARG_POINTS)
        point = JM_point_from_py(p)
        mupdf.pdf_add_annot_vertex(annot, point)
    mupdf.pdf_update_annot(annot)
    JM_add_annot_id(annot, 'A')
    return Annot(annot)