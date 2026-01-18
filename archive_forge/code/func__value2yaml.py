import argparse
import builtins
import enum
import importlib
import inspect
import io
import logging
import os
import pickle
import ply.lex
import re
import sys
import textwrap
import types
from operator import attrgetter
from pyomo.common.collections import Sequence, Mapping
from pyomo.common.deprecation import (
from pyomo.common.fileutils import import_file
from pyomo.common.formatting import wrap_reStructuredText
from pyomo.common.modeling import NOTSET
def _value2yaml(prefix, value, obj):
    _str = prefix
    if value is not None:
        try:
            _data = value._data if value is obj else value
            _str += _dump(_data, default_flow_style=True).rstrip()
            if _str.endswith('...'):
                _str = _str[:-3].rstrip()
        except:
            _str += str(type(_data))
    return _str.rstrip()