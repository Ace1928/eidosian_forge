import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def _unquote_match(s, pos):
    if s[:1] == '"' and s[-1:] == '"' or (s[:1] == "'" and s[-1:] == "'"):
        return (s[1:-1], pos + 1)
    else:
        return (s, pos)