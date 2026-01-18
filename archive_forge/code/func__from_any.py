from base64 import b64encode
from collections import namedtuple
import copy
import dataclasses
from functools import lru_cache
from io import BytesIO
import json
import logging
from numbers import Number
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Union
import matplotlib as mpl
from matplotlib import _api, _afm, cbook, ft2font
from matplotlib._fontconfig_pattern import (
from matplotlib.rcsetup import _validators
@classmethod
def _from_any(cls, arg):
    """
        Generic constructor which can build a `.FontProperties` from any of the
        following:

        - a `.FontProperties`: it is passed through as is;
        - `None`: a `.FontProperties` using rc values is used;
        - an `os.PathLike`: it is used as path to the font file;
        - a `str`: it is parsed as a fontconfig pattern;
        - a `dict`: it is passed as ``**kwargs`` to `.FontProperties`.
        """
    if arg is None:
        return cls()
    elif isinstance(arg, cls):
        return arg
    elif isinstance(arg, os.PathLike):
        return cls(fname=arg)
    elif isinstance(arg, str):
        return cls(arg)
    else:
        return cls(**arg)