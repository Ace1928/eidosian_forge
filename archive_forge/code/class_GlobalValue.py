import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
class GlobalValue(NamedValue, _ConstOpMixin, _HasMetadata):
    """
    A global value.
    """
    name_prefix = '@'
    deduplicate_name = False

    def __init__(self, *args, **kwargs):
        super(GlobalValue, self).__init__(*args, **kwargs)
        self.linkage = ''
        self.storage_class = ''
        self.section = ''
        self.metadata = {}