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
class ParallelThreadsAvailableNode(AtomicExprNode):
    """
    Note: this is disabled and not a valid directive at this moment

    Implements cython.parallel.threadsavailable(). If we are called from the
    sequential part of the application, we need to call omp_get_max_threads(),
    and in the parallel part we can just call omp_get_num_threads()
    """
    type = PyrexTypes.c_int_type

    def analyse_types(self, env):
        self.is_temp = True
        return self

    def generate_result_code(self, code):
        code.putln('#ifdef _OPENMP')
        code.putln('if (omp_in_parallel()) %s = omp_get_max_threads();' % self.temp_code)
        code.putln('else %s = omp_get_num_threads();' % self.temp_code)
        code.putln('#else')
        code.putln('%s = 1;' % self.temp_code)
        code.putln('#endif')

    def result(self):
        return self.temp_code