import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
class MetaDataArgument(_StrCaching, _StringReferenceCaching, Value):
    """
    An argument value to a function taking metadata arguments.
    This can wrap any other kind of LLVM value.

    Do not instantiate directly, Builder.call() will create these
    automatically.
    """

    def __init__(self, value):
        assert isinstance(value, Value)
        assert not isinstance(value.type, types.MetaDataType)
        self.type = types.MetaDataType()
        self.wrapped_value = value

    def _get_reference(self):
        return '{0} {1}'.format(self.wrapped_value.type, self.wrapped_value.get_reference())
    _to_string = _get_reference