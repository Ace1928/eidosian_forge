import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def _str_section(self, name):
    out = []
    if self[name]:
        out += self._str_header(name)
        out += self[name]
        out += ['']
    return out