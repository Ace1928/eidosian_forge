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
class SliceIntNode(SliceNode):
    is_temp = 0

    def calculate_constant_result(self):
        self.constant_result = slice(self.start.constant_result, self.stop.constant_result, self.step.constant_result)

    def compile_time_value(self, denv):
        start = self.start.compile_time_value(denv)
        stop = self.stop.compile_time_value(denv)
        step = self.step.compile_time_value(denv)
        try:
            return slice(start, stop, step)
        except Exception as e:
            self.compile_time_value_error(e)

    def may_be_none(self):
        return False

    def analyse_types(self, env):
        self.start = self.start.analyse_types(env)
        self.stop = self.stop.analyse_types(env)
        self.step = self.step.analyse_types(env)
        if not self.start.is_none:
            self.start = self.start.coerce_to_integer(env)
        if not self.stop.is_none:
            self.stop = self.stop.coerce_to_integer(env)
        if not self.step.is_none:
            self.step = self.step.coerce_to_integer(env)
        if self.start.is_literal and self.stop.is_literal and self.step.is_literal:
            self.is_literal = True
            self.is_temp = False
        return self

    def calculate_result_code(self):
        pass

    def generate_result_code(self, code):
        for a in (self.start, self.stop, self.step):
            if isinstance(a, CloneNode):
                a.arg.result()