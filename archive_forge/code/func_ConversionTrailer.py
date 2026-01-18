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
def ConversionTrailer(i: str):
    t = i.lower()
    text = ''
    json = ']\n}'
    html = '</body>\n</html>\n'
    xml = '</document>\n'
    xhtml = html
    if t == 'html':
        r = html
    elif t == 'json':
        r = json
    elif t == 'xml':
        r = xml
    elif t == 'xhtml':
        r = xhtml
    else:
        r = text
    return r