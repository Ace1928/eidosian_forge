import copy
import datetime
import decimal
import types
import warnings
from functools import reduce
from zope.interface import implementer
from incremental import Version
from twisted.persisted.crefutil import (
from twisted.python.compat import nativeString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.reflect import namedAny, namedObject, qual
from twisted.spread.interfaces import IJellyable, IUnjellyable
class SecurityOptions:
    """
    This will by default disallow everything, except for 'none'.
    """
    basicTypes = ['dictionary', 'list', 'tuple', 'reference', 'dereference', 'unpersistable', 'persistent', 'long_int', 'long', 'dict']

    def __init__(self):
        """
        SecurityOptions() initialize.
        """
        self.allowedTypes = {b'None': 1, b'bool': 1, b'boolean': 1, b'string': 1, b'str': 1, b'int': 1, b'float': 1, b'datetime': 1, b'time': 1, b'date': 1, b'timedelta': 1, b'NoneType': 1, b'unicode': 1, b'decimal': 1, b'set': 1, b'frozenset': 1}
        self.allowedModules = {}
        self.allowedClasses = {}

    def allowBasicTypes(self):
        """
        Allow all `basic' types.  (Dictionary and list.  Int, string, and float
        are implicitly allowed.)
        """
        self.allowTypes(*self.basicTypes)

    def allowTypes(self, *types):
        """
        SecurityOptions.allowTypes(typeString): Allow a particular type, by its
        name.
        """
        for typ in types:
            if isinstance(typ, str):
                typ = typ.encode('utf-8')
            if not isinstance(typ, bytes):
                typ = qual(typ)
            self.allowedTypes[typ] = 1

    def allowInstancesOf(self, *classes):
        """
        SecurityOptions.allowInstances(klass, klass, ...): allow instances
        of the specified classes

        This will also allow the 'instance', 'class' (renamed 'classobj' in
        Python 2.3), and 'module' types, as well as basic types.
        """
        self.allowBasicTypes()
        self.allowTypes('instance', 'class', 'classobj', 'module')
        for klass in classes:
            self.allowTypes(qual(klass))
            self.allowModules(klass.__module__)
            self.allowedClasses[klass] = 1

    def allowModules(self, *modules):
        """
        SecurityOptions.allowModules(module, module, ...): allow modules by
        name. This will also allow the 'module' type.
        """
        for module in modules:
            if type(module) == types.ModuleType:
                module = module.__name__
            if not isinstance(module, bytes):
                module = module.encode('utf-8')
            self.allowedModules[module] = 1

    def isModuleAllowed(self, moduleName):
        """
        SecurityOptions.isModuleAllowed(moduleName) -> boolean
        returns 1 if a module by that name is allowed, 0 otherwise
        """
        if not isinstance(moduleName, bytes):
            moduleName = moduleName.encode('utf-8')
        return moduleName in self.allowedModules

    def isClassAllowed(self, klass):
        """
        SecurityOptions.isClassAllowed(class) -> boolean
        Assumes the module has already been allowed.  Returns 1 if the given
        class is allowed, 0 otherwise.
        """
        return klass in self.allowedClasses

    def isTypeAllowed(self, typeName):
        """
        SecurityOptions.isTypeAllowed(typeName) -> boolean
        Returns 1 if the given type is allowed, 0 otherwise.
        """
        if not isinstance(typeName, bytes):
            typeName = typeName.encode('utf-8')
        return typeName in self.allowedTypes or b'.' in typeName