import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
class ProxyClass(ProxyBase):
    flags = 0

    def __init__(self, objectname, is_attribute, args=(), noInstantiation=False):
        if objectname:
            if is_attribute:
                objectname = 'self.' + objectname
            self._uic_name = objectname
        else:
            self._uic_name = 'Unnamed'
        if not noInstantiation:
            funcall = '%s(%s)' % (moduleMember(self.module, self.__class__.__name__), ', '.join(map(str, args)))
            if objectname:
                funcall = '%s = %s' % (objectname, funcall)
            write_code(funcall)

    def __str__(self):
        return self._uic_name

    def __getattribute__(self, attribute):
        try:
            return object.__getattribute__(self, attribute)
        except AttributeError:
            return ProxyClassMember(self, attribute, self.flags)