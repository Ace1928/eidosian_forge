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
def JM_image_filter(opaque, ctm, name, image):
    assert isinstance(ctm, mupdf.FzMatrix)
    r = mupdf.FzRect(mupdf.FzRect.Fixed_UNIT)
    q = mupdf.fz_transform_quad(mupdf.fz_quad_from_rect(r), ctm)
    q = mupdf.fz_transform_quad(q, g_img_info_matrix)
    temp = (name, JM_py_from_quad(q))
    g_img_info.append(temp)