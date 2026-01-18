from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
class PropertyScope(Scope):
    is_property_scope = 1

    def __init__(self, name, class_scope):
        outer_scope = class_scope.global_scope() if class_scope.outer_scope else None
        Scope.__init__(self, name, outer_scope, parent_scope=class_scope)
        self.parent_type = class_scope.parent_type
        self.directives = class_scope.directives

    def declare_cfunction(self, name, type, pos, *args, **kwargs):
        """Declare a C property function.
        """
        if type.return_type.is_void:
            error(pos, "C property method cannot return 'void'")
        if type.args and type.args[0].type is py_object_type:
            type.args[0].type = self.parent_scope.parent_type
        elif len(type.args) != 1:
            error(pos, 'C property method must have a single (self) argument')
        elif not (type.args[0].type.is_pyobject or type.args[0].type is self.parent_scope.parent_type):
            error(pos, 'C property method must have a single (object) argument')
        entry = Scope.declare_cfunction(self, name, type, pos, *args, **kwargs)
        entry.is_cproperty = True
        return entry

    def declare_pyfunction(self, name, pos, allow_redefine=False):
        signature = get_property_accessor_signature(name)
        if signature:
            entry = self.declare(name, name, py_object_type, pos, 'private')
            entry.is_special = 1
            entry.signature = signature
            return entry
        else:
            error(pos, 'Only __get__, __set__ and __del__ methods allowed in a property declaration')
            return None