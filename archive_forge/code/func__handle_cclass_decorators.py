from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
def _handle_cclass_decorators(self, env):
    extra_directives = {}
    if not self.decorators:
        return extra_directives
    from . import ExprNodes
    remaining_decorators = []
    for original_decorator in self.decorators:
        decorator = original_decorator.decorator
        decorator_call = None
        if isinstance(decorator, ExprNodes.CallNode):
            decorator_call = decorator
            decorator = decorator.function
        known_name = Builtin.exprnode_to_known_standard_library_name(decorator, env)
        if known_name == 'functools.total_ordering':
            if decorator_call:
                error(decorator_call.pos, 'total_ordering cannot be called.')
            extra_directives['total_ordering'] = True
            continue
        elif known_name == 'dataclasses.dataclass':
            args = None
            kwds = {}
            if decorator_call:
                if isinstance(decorator_call, ExprNodes.SimpleCallNode):
                    args = decorator_call.args
                else:
                    args = decorator_call.positional_args.args
                    kwds_ = decorator_call.keyword_args
                    if kwds_:
                        kwds = kwds_.as_python_dict()
            extra_directives[known_name] = (args, kwds)
            continue
        remaining_decorators.append(original_decorator)
    if remaining_decorators:
        error(remaining_decorators[0].pos, 'Cdef functions/classes cannot take arbitrary decorators.')
    self.decorators = remaining_decorators
    return extra_directives