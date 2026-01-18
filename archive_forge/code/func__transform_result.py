import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def _transform_result(typ, result):
    """Convert the result back into the input type.
    """
    if issubclass(typ, bytes):
        return tostring(result, encoding='utf-8')
    elif issubclass(typ, str):
        return tostring(result, encoding='unicode')
    else:
        return result