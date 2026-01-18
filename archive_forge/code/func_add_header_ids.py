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
def add_header_ids(self):
    """
        Look for `<h1..6>` items in `self` and adds unique `id`
        attributes if not already present.
        """
    dom = self.body
    i = 0
    x = dom.find(None, None, None)
    while x:
        name = x.tagname
        if len(name) == 2 and name[0] == 'h' and (name[1] in '123456'):
            attr = x.get_attribute_value('id')
            if not attr:
                id_ = f'h_id_{i}'
                x.set_attribute('id', id_)
                i += 1
        x = x.find_next(None, None, None)