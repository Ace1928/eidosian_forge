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
def add_caret_annot(self, point: point_like) -> Annot:
    """Add a 'Caret' annotation."""
    old_rotation = annot_preprocess(self)
    try:
        annot = self._add_caret_annot(point)
    finally:
        if old_rotation != 0:
            self.set_rotation(old_rotation)
    annot = Annot(annot)
    annot_postprocess(self, annot)
    assert hasattr(annot, 'parent')
    return annot