import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
class LiteralProxyClass(ProxyClass):
    """LiteralObject(*args) -> new literal class

    a literal class can be used as argument in a function call

    >>> class Foo(LiteralProxyClass): pass
    >>> str(Foo(1,2,3)) == "Foo(1,2,3)"
    """
    flags = AS_ARGUMENT

    def __init__(self, *args):
        self._uic_name = '%s(%s)' % (moduleMember(self.module, self.__class__.__name__), ', '.join(map(as_string, args)))