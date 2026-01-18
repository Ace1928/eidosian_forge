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
def JM_embedded_clean(pdf):
    """
    perform some cleaning if we have /EmbeddedFiles:
    (1) remove any /Limits if /Names exists
    (2) remove any empty /Collection
    (3) set /PageMode/UseAttachments
    """
    root = mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('Root'))
    coll = mupdf.pdf_dict_get(root, PDF_NAME('Collection'))
    if coll.m_internal and mupdf.pdf_dict_len(coll) == 0:
        mupdf.pdf_dict_del(root, PDF_NAME('Collection'))
    efiles = mupdf.pdf_dict_getl(root, PDF_NAME('Names'), PDF_NAME('EmbeddedFiles'), PDF_NAME('Names'))
    if efiles.m_internal:
        mupdf.pdf_dict_put_name(root, PDF_NAME('PageMode'), 'UseAttachments')