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
def _selected_int_kind_func(r):
    m = 10 ** r
    if m <= 2 ** 8:
        return 1
    if m <= 2 ** 16:
        return 2
    if m <= 2 ** 32:
        return 4
    if m <= 2 ** 63:
        return 8
    if m <= 2 ** 128:
        return 16
    return -1