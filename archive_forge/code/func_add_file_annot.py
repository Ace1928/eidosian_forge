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
def add_file_annot(self, point: point_like, buffer_: typing.ByteString, filename: str, ufilename: OptStr=None, desc: OptStr=None, icon: OptStr=None) -> Annot:
    """Add a 'FileAttachment' annotation."""
    old_rotation = annot_preprocess(self)
    try:
        annot = self._add_file_annot(point, buffer_, filename, ufilename=ufilename, desc=desc, icon=icon)
    finally:
        if old_rotation != 0:
            self.set_rotation(old_rotation)
    annot_postprocess(self, annot)
    return annot