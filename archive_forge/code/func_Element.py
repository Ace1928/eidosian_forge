import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def Element(*args, **kw):
    """Create a new HTML Element.

    This can also be used for XHTML documents.
    """
    v = html_parser.makeelement(*args, **kw)
    return v