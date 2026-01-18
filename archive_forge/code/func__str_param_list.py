import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def _str_param_list(self, name):
    out = []
    if self[name]:
        out += self._str_header(name)
        for param in self[name]:
            parts = []
            if param.name:
                parts.append(param.name)
            if param.type:
                parts.append(param.type)
            out += [' : '.join(parts)]
            if param.desc and ''.join(param.desc).strip():
                out += self._str_indent(param.desc)
        out += ['']
    return out