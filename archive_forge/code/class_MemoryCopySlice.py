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
class MemoryCopySlice(MemoryCopyNode):
    """
    Copy the contents of slice src to slice dst. Does not support indirect
    slices.

        memslice1[...] = memslice2
        memslice1[:] = memslice2
    """
    is_memview_copy_assignment = True
    copy_slice_cname = '__pyx_memoryview_copy_contents'

    def _generate_assignment_code(self, src, code):
        dst = self.dst
        src.type.assert_direct_dims(src.pos)
        dst.type.assert_direct_dims(dst.pos)
        code.putln(code.error_goto_if_neg('%s(%s, %s, %d, %d, %d)' % (self.copy_slice_cname, src.result(), dst.result(), src.type.ndim, dst.type.ndim, dst.type.dtype.is_pyobject), dst.pos))