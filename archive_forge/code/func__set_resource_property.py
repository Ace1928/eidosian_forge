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
def _set_resource_property(self, name, xref):
    page = self._pdf_page()
    ASSERT_PDF(page)
    JM_set_resource_property(page.obj(), name, xref)