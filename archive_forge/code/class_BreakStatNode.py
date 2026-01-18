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
class BreakStatNode(StatNode):
    child_attrs = []
    is_terminator = True

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        if not code.break_label:
            error(self.pos, 'break statement not inside loop')
        else:
            code.put_goto(code.break_label)