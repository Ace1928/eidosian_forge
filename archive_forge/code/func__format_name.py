import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
def _format_name(self):
    name = self.name
    if not _SIMPLE_IDENTIFIER_RE.match(name):
        name = name.replace('\\', '\\5c').replace('"', '\\22')
        name = '"{0}"'.format(name)
    return name