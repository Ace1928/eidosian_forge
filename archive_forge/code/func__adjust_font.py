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
def _adjust_font(self):
    """Ensure text_font is from our list and correctly spelled.
        """
    if not self.text_font:
        self.text_font = 'Helv'
        return
    valid_fonts = ('Cour', 'TiRo', 'Helv', 'ZaDb')
    for f in valid_fonts:
        if self.text_font.lower() == f.lower():
            self.text_font = f
            return
    self.text_font = 'Helv'
    return