import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
def _unikey_or_keysym_to_mplkey(unikey, keysym):
    """
    Convert a Unicode key or X keysym to a Matplotlib key name.

    The Unicode key is checked first; this avoids having to list most printable
    keysyms such as ``EuroSign``.
    """
    if unikey and unikey.isprintable():
        return unikey
    key = keysym.lower()
    if key.startswith('kp_'):
        key = key[3:]
    if key.startswith('page_'):
        key = key.replace('page_', 'page')
    if key.endswith(('_l', '_r')):
        key = key[:-2]
    if sys.platform == 'darwin' and key == 'meta':
        key = 'cmd'
    key = {'return': 'enter', 'prior': 'pageup', 'next': 'pagedown'}.get(key, key)
    return key