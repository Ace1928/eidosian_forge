import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def _ensure_exprdict(r):
    if isinstance(r, int):
        return {'typespec': 'integer'}
    if isinstance(r, float):
        return {'typespec': 'real'}
    if isinstance(r, complex):
        return {'typespec': 'complex'}
    if isinstance(r, dict):
        return r
    raise AssertionError(repr(r))