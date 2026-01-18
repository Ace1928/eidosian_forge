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
def _is_visit_pair(obj):
    return isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], (int, str))