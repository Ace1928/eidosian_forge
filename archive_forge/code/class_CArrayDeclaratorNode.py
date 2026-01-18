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
class CArrayDeclaratorNode(CDeclaratorNode):
    child_attrs = ['base', 'dimension']

    def analyse(self, base_type, env, nonempty=0, visibility=None, in_pxd=False):
        if base_type.is_cpp_class and base_type.is_template_type() or base_type.is_cfunction or base_type.python_type_constructor_name:
            from .ExprNodes import TupleNode
            if isinstance(self.dimension, TupleNode):
                args = self.dimension.args
            else:
                args = (self.dimension,)
            values = [v.analyse_as_type(env) for v in args]
            if None in values:
                ix = values.index(None)
                error(args[ix].pos, 'Template parameter not a type')
                base_type = error_type
            else:
                base_type = base_type.specialize_here(self.pos, env, values)
            return self.base.analyse(base_type, env, nonempty=nonempty, visibility=visibility, in_pxd=in_pxd)
        if self.dimension:
            self.dimension = self.dimension.analyse_const_expression(env)
            if not self.dimension.type.is_int:
                error(self.dimension.pos, 'Array dimension not integer')
            size = self.dimension.get_constant_c_result_code()
            if size is not None:
                try:
                    size = int(size)
                except ValueError:
                    pass
        else:
            size = None
        if not base_type.is_complete():
            error(self.pos, "Array element type '%s' is incomplete" % base_type)
        if base_type.is_pyobject:
            error(self.pos, 'Array element cannot be a Python object')
        if base_type.is_cfunction:
            error(self.pos, 'Array element cannot be a function')
        array_type = PyrexTypes.c_array_type(base_type, size)
        return self.base.analyse(array_type, env, nonempty=nonempty, visibility=visibility, in_pxd=in_pxd)