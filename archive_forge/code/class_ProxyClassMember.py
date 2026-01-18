import sys
import re
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.misc import Literal, moduleMember
class ProxyClassMember(object):

    def __init__(self, proxy, function_name, flags):
        self.proxy = proxy
        self.function_name = function_name
        self.flags = flags

    def __str__(self):
        return '%s.%s' % (self.proxy, self.function_name)

    def __call__(self, *args):
        func_call = '%s.%s(%s)' % (self.proxy, self.function_name, ', '.join(map(as_string, args)))
        if self.flags & AS_ARGUMENT:
            self.proxy._uic_name = func_call
            return self.proxy
        else:
            needs_translation = False
            for arg in args:
                if isinstance(arg, i18n_string):
                    needs_translation = True
            if needs_translation:
                i18n_print(func_call)
            else:
                write_code(func_call)