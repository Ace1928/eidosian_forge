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
class MemoryCopyScalar(MemoryCopyNode):
    """
    Assign a scalar to a slice. dst must be simple, scalar will be assigned
    to a correct type and not just something assignable.

        memslice1[...] = 0.0
        memslice1[:] = 0.0
    """

    def __init__(self, pos, dst):
        super(MemoryCopyScalar, self).__init__(pos, dst)
        self.type = dst.type.dtype

    def _generate_assignment_code(self, scalar, code):
        from . import MemoryView
        self.dst.type.assert_direct_dims(self.dst.pos)
        dtype = self.dst.type.dtype
        type_decl = dtype.declaration_code('')
        slice_decl = self.dst.type.declaration_code('')
        code.begin_block()
        code.putln('%s __pyx_temp_scalar = %s;' % (type_decl, scalar.result()))
        if self.dst.result_in_temp() or self.dst.is_simple():
            dst_temp = self.dst.result()
        else:
            code.putln('%s __pyx_temp_slice = %s;' % (slice_decl, self.dst.result()))
            dst_temp = '__pyx_temp_slice'
        force_strided = False
        indices = self.dst.original_indices
        for idx in indices:
            if isinstance(idx, SliceNode) and (not (idx.start.is_none and idx.stop.is_none and idx.step.is_none)):
                force_strided = True
        slice_iter_obj = MemoryView.slice_iter(self.dst.type, dst_temp, self.dst.type.ndim, code, force_strided=force_strided)
        p = slice_iter_obj.start_loops()
        if dtype.is_pyobject:
            code.putln('Py_DECREF(*(PyObject **) %s);' % p)
        code.putln('*((%s *) %s) = __pyx_temp_scalar;' % (type_decl, p))
        if dtype.is_pyobject:
            code.putln('Py_INCREF(__pyx_temp_scalar);')
        slice_iter_obj.end_loops()
        code.end_block()