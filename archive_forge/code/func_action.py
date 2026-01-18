import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
@action.deleter
def action(self):
    attrib = self.attrib
    if 'action' in attrib:
        del attrib['action']