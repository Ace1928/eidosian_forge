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
def JM_expand_fname(name):
    """
    Make /DA string of annotation
    """
    if not name:
        return 'Helv'
    if name.startswith('Co'):
        return 'Cour'
    if name.startswith('co'):
        return 'Cour'
    if name.startswith('Ti'):
        return 'TiRo'
    if name.startswith('ti'):
        return 'TiRo'
    if name.startswith('Sy'):
        return 'Symb'
    if name.startswith('sy'):
        return 'Symb'
    if name.startswith('Za'):
        return 'ZaDb'
    if name.startswith('za'):
        return 'ZaDb'
    return 'Helv'