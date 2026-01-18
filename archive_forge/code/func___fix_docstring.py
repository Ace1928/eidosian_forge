import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def __fix_docstring(s):
    if not s:
        return s
    sub = re.compile("^(\\s*)u'", re.M).sub
    return sub("\\1'", s)