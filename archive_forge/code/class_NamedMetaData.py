import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
class NamedMetaData(object):
    """
    A named metadata node.

    Do not instantiate directly, use Module.add_named_metadata() instead.
    """

    def __init__(self, parent):
        self.parent = parent
        self.operands = []

    def add(self, md):
        self.operands.append(md)