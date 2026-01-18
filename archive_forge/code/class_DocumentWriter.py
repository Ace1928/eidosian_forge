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
class DocumentWriter:

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __init__(self, path, options=''):
        if isinstance(path, str):
            pass
        elif hasattr(path, 'absolute'):
            path = str(path)
        elif hasattr(path, 'name'):
            path = path.name
        if isinstance(path, str):
            self.this = mupdf.FzDocumentWriter(path, options, mupdf.FzDocumentWriter.PathType_PDF)
        else:
            out = JM_new_output_fileptr(path)
            self.this = mupdf.FzDocumentWriter(out, options, mupdf.FzDocumentWriter.OutputType_PDF)
            assert out.m_internal_value() == 0
            assert hasattr(self.this, '_out')

    def begin_page(self, mediabox):
        mediabox2 = JM_rect_from_py(mediabox)
        device = mupdf.fz_begin_page(self.this, mediabox2)
        device_wrapper = DeviceWrapper(device)
        return device_wrapper

    def close(self):
        mupdf.fz_close_document_writer(self.this)

    def end_page(self):
        mupdf.fz_end_page(self.this)