from __future__ import annotations
import ast
import base64
import datetime as dt
import json
import logging
import numbers
import os
import pathlib
import re
import sys
import urllib.parse as urlparse
from collections import OrderedDict, defaultdict
from collections.abc import MutableMapping, MutableSequence
from datetime import datetime
from functools import partial
from html import escape  # noqa
from importlib import import_module
from typing import Any, AnyStr
import bokeh
import numpy as np
import param
from bokeh.core.has_props import _default_resolver
from bokeh.model import Model
from packaging.version import Version
from .checks import (  # noqa
from .parameters import (  # noqa
def abbreviated_repr(value, max_length=25, natural_breaks=(',', ' ')):
    """
    Returns an abbreviated repr for the supplied object. Attempts to
    find a natural break point while adhering to the maximum length.
    """
    if isinstance(value, list):
        vrepr = '[' + ', '.join([abbreviated_repr(v) for v in value]) + ']'
    if isinstance(value, param.Parameterized):
        vrepr = type(value).__name__
    else:
        vrepr = repr(value)
    if len(vrepr) > max_length:
        abbrev = vrepr[max_length // 2:]
        natural_break = None
        for brk in natural_breaks:
            if brk in abbrev:
                natural_break = abbrev.index(brk) + max_length // 2
                break
        if natural_break and natural_break < max_length:
            max_length = natural_break + 1
        end_char = ''
        if isinstance(value, list):
            end_char = ']'
        elif isinstance(value, OrderedDict):
            end_char = '])'
        elif isinstance(value, (dict, set)):
            end_char = '}'
        return vrepr[:max_length + 1] + '...' + end_char
    return vrepr