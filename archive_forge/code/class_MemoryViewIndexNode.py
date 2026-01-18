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
class MemoryViewIndexNode(BufferIndexNode):
    is_memview_index = True
    is_buffer_access = False

    def analyse_types(self, env, getting=True):
        from . import MemoryView
        self.is_pythran_mode = has_np_pythran(env)
        indices = self.indices
        have_slices, indices, newaxes = MemoryView.unellipsify(indices, self.base.type.ndim)
        if not getting:
            self.writable_needed = True
            if self.base.is_name or self.base.is_attribute:
                self.base.entry.type.writable_needed = True
        self.memslice_index = not newaxes and len(indices) == self.base.type.ndim
        axes = []
        index_type = PyrexTypes.c_py_ssize_t_type
        new_indices = []
        if len(indices) - len(newaxes) > self.base.type.ndim:
            self.type = error_type
            error(indices[self.base.type.ndim].pos, 'Too many indices specified for type %s' % self.base.type)
            return self
        axis_idx = 0
        for i, index in enumerate(indices[:]):
            index = index.analyse_types(env)
            if index.is_none:
                self.is_memview_slice = True
                new_indices.append(index)
                axes.append(('direct', 'strided'))
                continue
            access, packing = self.base.type.axes[axis_idx]
            axis_idx += 1
            if index.is_slice:
                self.is_memview_slice = True
                if index.step.is_none:
                    axes.append((access, packing))
                else:
                    axes.append((access, 'strided'))
                for attr in ('start', 'stop', 'step'):
                    value = getattr(index, attr)
                    if not value.is_none:
                        value = value.coerce_to(index_type, env)
                        setattr(index, attr, value)
                        new_indices.append(value)
            elif index.type.is_int or index.type.is_pyobject:
                if index.type.is_pyobject:
                    performance_hint(index.pos, 'Index should be typed for more efficient access', env)
                self.is_memview_index = True
                index = index.coerce_to(index_type, env)
                indices[i] = index
                new_indices.append(index)
            else:
                self.type = error_type
                error(index.pos, 'Invalid index for memoryview specified, type %s' % index.type)
                return self
        self.is_memview_index = self.is_memview_index and (not self.is_memview_slice)
        self.indices = new_indices
        self.original_indices = indices
        self.nogil = env.nogil
        self.analyse_operation(env, getting, axes)
        self.wrap_in_nonecheck_node(env)
        return self

    def analyse_operation(self, env, getting, axes):
        self.none_error_message = 'Cannot index None memoryview slice'
        self.analyse_buffer_index(env, getting)

    def analyse_broadcast_operation(self, rhs):
        """
        Support broadcasting for slice assignment.
        E.g.
            m_2d[...] = m_1d  # or,
            m_1d[...] = m_2d  # if the leading dimension has extent 1
        """
        if self.type.is_memoryviewslice:
            lhs = self
            if lhs.is_memview_broadcast or rhs.is_memview_broadcast:
                lhs.is_memview_broadcast = True
                rhs.is_memview_broadcast = True

    def analyse_as_memview_scalar_assignment(self, rhs):
        lhs = self.analyse_assignment(rhs)
        if lhs:
            rhs.is_memview_copy_assignment = lhs.is_memview_copy_assignment
            return lhs
        return self