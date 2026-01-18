import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
class ReturnValue(_BaseArgument):
    """
    The specification of a function's return value.
    """

    def __str__(self):
        attrs = self.attributes._to_list(self.type)
        if attrs:
            return '{0} {1}'.format(' '.join(attrs), self.type)
        else:
            return str(self.type)