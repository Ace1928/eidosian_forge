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
class ClosureScope(LocalScope):
    is_closure_scope = True

    def __init__(self, name, scope_name, outer_scope, parent_scope=None):
        LocalScope.__init__(self, name, outer_scope, parent_scope)
        self.closure_cname = '%s%s' % (Naming.closure_scope_prefix, scope_name)

    def declare_pyfunction(self, name, pos, allow_redefine=False):
        return LocalScope.declare_pyfunction(self, name, pos, allow_redefine, visibility='private')

    def declare_assignment_expression_target(self, name, type, pos):
        return self.declare_var(name, type, pos)