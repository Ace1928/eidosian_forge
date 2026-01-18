import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
class GlobalVariable(GlobalValue):
    """
    A global variable.
    """

    def __init__(self, module, typ, name, addrspace=0):
        assert isinstance(typ, types.Type)
        super(GlobalVariable, self).__init__(module, typ.as_pointer(addrspace), name=name)
        self.value_type = typ
        self.initializer = None
        self.unnamed_addr = False
        self.global_constant = False
        self.addrspace = addrspace
        self.align = None
        self.parent.add_global(self)

    def descr(self, buf):
        if self.global_constant:
            kind = 'constant'
        else:
            kind = 'global'
        if not self.linkage:
            linkage = 'external' if self.initializer is None else ''
        else:
            linkage = self.linkage
        if linkage:
            buf.append(linkage + ' ')
        if self.storage_class:
            buf.append(self.storage_class + ' ')
        if self.unnamed_addr:
            buf.append('unnamed_addr ')
        if self.addrspace != 0:
            buf.append('addrspace({0:d}) '.format(self.addrspace))
        buf.append('{kind} {type}'.format(kind=kind, type=self.value_type))
        if self.initializer is not None:
            if self.initializer.type != self.value_type:
                raise TypeError('got initializer of type %s for global value type %s' % (self.initializer.type, self.value_type))
            buf.append(' ' + self.initializer.get_reference())
        elif linkage not in ('external', 'extern_weak'):
            buf.append(' ' + self.value_type(Undefined).get_reference())
        if self.section:
            buf.append(', section "%s"' % (self.section,))
        if self.align is not None:
            buf.append(', align %d' % (self.align,))
        if self.metadata:
            buf.append(self._stringify_metadata(leading_comma=True))
        buf.append('\n')