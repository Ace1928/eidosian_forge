from __future__ import print_function, absolute_import
import sys
from shibokensupport.signature import inspect
from shibokensupport.signature import get_signature
class ExactEnumerator(object):
    """
    ExactEnumerator enumerates all signatures in a module as they are.

    This class is used for generating complete listings of all signatures.
    An appropriate formatter should be supplied, if printable output
    is desired.
    """

    def __init__(self, formatter, result_type=dict):
        self.fmt = formatter
        self.result_type = result_type
        self.fmt.level = 0
        self.fmt.after_enum = self.after_enum
        self._after_enum = False

    def after_enum(self):
        ret = self._after_enum
        self._after_enum = False

    def module(self, mod_name):
        __import__(mod_name)
        with self.fmt.module(mod_name):
            module = sys.modules[mod_name]
            members = inspect.getmembers(module, inspect.isclass)
            functions = inspect.getmembers(module, inspect.isroutine)
            ret = self.result_type()
            self.fmt.class_name = None
            for class_name, klass in members:
                ret.update(self.klass(class_name, klass))
            if isinstance(klass, EnumType):
                self.enum(klass)
            for func_name, func in functions:
                ret.update(self.function(func_name, func))
            return ret

    def klass(self, class_name, klass):
        bases_list = []
        for base in klass.__bases__:
            name = base.__name__
            if name in ('object', 'type'):
                pass
            else:
                modname = base.__module__
                name = modname + '.' + base.__name__
            bases_list.append(name)
        class_str = '{}({})'.format(class_name, ', '.join(bases_list))
        with self.fmt.klass(class_name, class_str):
            ret = self.result_type()
            class_members = sorted(list(klass.__dict__.items()))
            subclasses = []
            functions = []
            for thing_name, thing in class_members:
                if inspect.isclass(thing):
                    subclass_name = '.'.join((class_name, thing_name))
                    subclasses.append((subclass_name, thing))
                elif inspect.isroutine(thing):
                    func_name = thing_name.split('.')[0]
                    functions.append((func_name, thing))
            self.fmt.level += 1
            for subclass_name, subclass in subclasses:
                ret.update(self.klass(subclass_name, subclass))
                if isinstance(subclass, EnumType):
                    self.enum(subclass)
            ret = self.function('__init__', klass)
            for func_name, func in functions:
                func_kind = get_signature(func, '__func_kind__')
                modifier = func_kind if func_kind in ('staticmethod', 'classmethod') else None
                ret.update(self.function(func_name, func, modifier))
            self.fmt.level -= 1
        return ret

    def function(self, func_name, func, modifier=None):
        self.fmt.level += 1
        ret = self.result_type()
        signature = getattr(func, '__signature__', None)
        if signature is not None:
            with self.fmt.function(func_name, signature, modifier) as key:
                ret[key] = signature
        self.fmt.level -= 1
        return ret

    def enum(self, subclass):
        if not hasattr(self.fmt, 'enum'):
            return
        class_name = subclass.__name__
        for enum_name, value in subclass.__dict__.items():
            if type(type(value)) is EnumType:
                with self.fmt.enum(class_name, enum_name, int(value)):
                    pass
        self._after_enum = True