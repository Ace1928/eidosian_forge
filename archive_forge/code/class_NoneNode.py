from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
class NoneNode(PyConstNode):
    is_none = 1
    value = 'Py_None'
    constant_result = None

    def compile_time_value(self, denv):
        return None

    def may_be_none(self):
        return True

    def coerce_to(self, dst_type, env):
        if not (dst_type.is_pyobject or dst_type.is_memoryviewslice or dst_type.is_error):
            error(self.pos, 'Cannot assign None to %s' % dst_type)
        return super(NoneNode, self).coerce_to(dst_type, env)