import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
class FunctionAttributes(AttributeSet):
    _known = frozenset(['argmemonly', 'alwaysinline', 'builtin', 'cold', 'convergent', 'inaccessiblememonly', 'inaccessiblemem_or_argmemonly', 'inlinehint', 'jumptable', 'minsize', 'naked', 'nobuiltin', 'noduplicate', 'noimplicitfloat', 'noinline', 'nonlazybind', 'norecurse', 'noredzone', 'noreturn', 'nounwind', 'optnone', 'optsize', 'readnone', 'readonly', 'returns_twice', 'sanitize_address', 'sanitize_memory', 'sanitize_thread', 'ssp', 'sspreg', 'sspstrong', 'uwtable'])

    def __init__(self, args=()):
        self._alignstack = 0
        self._personality = None
        super(FunctionAttributes, self).__init__(args)

    def add(self, name):
        if name == 'alwaysinline' and 'noinline' in self or (name == 'noinline' and 'alwaysinline' in self):
            raise ValueError("Can't have alwaysinline and noinline")
        super().add(name)

    @property
    def alignstack(self):
        return self._alignstack

    @alignstack.setter
    def alignstack(self, val):
        assert val >= 0
        self._alignstack = val

    @property
    def personality(self):
        return self._personality

    @personality.setter
    def personality(self, val):
        assert val is None or isinstance(val, GlobalValue)
        self._personality = val

    def _to_list(self, ret_type):
        attrs = super()._to_list(ret_type)
        if self.alignstack:
            attrs.append('alignstack({0:d})'.format(self.alignstack))
        if self.personality:
            attrs.append('personality {persty} {persfn}'.format(persty=self.personality.type, persfn=self.personality.get_reference()))
        return attrs