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
def _eval_length(length, params):
    if length in ['(:)', '(*)', '*']:
        return '(*)'
    return _eval_scalar(length, params)