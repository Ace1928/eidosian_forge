import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def _asStringList(self, sep=''):
    out = []
    for item in self.__toklist:
        if out and sep:
            out.append(sep)
        if isinstance(item, ParseResults):
            out += item._asStringList()
        else:
            out.append(_ustr(item))
    return out