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
def _makePixmap(self, doc, ctm, cs, alpha=0, annots=1, clip=None):
    pix = JM_pixmap_from_page(doc, self.this, ctm, cs, alpha, annots, clip)
    return Pixmap(pix)