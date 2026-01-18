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
class TemplatedTypeNode(CBaseTypeNode):
    child_attrs = ['base_type_node', 'positional_args', 'keyword_args', 'dtype_node']
    is_templated_type_node = True
    dtype_node = None
    name = None

    def _analyse_template_types(self, env, base_type):
        require_optional_types = base_type.python_type_constructor_name == 'typing.Optional'
        require_python_types = base_type.python_type_constructor_name == 'dataclasses.ClassVar'
        in_c_type_context = env.in_c_type_context and (not require_python_types)
        template_types = []
        for template_node in self.positional_args:
            with env.new_c_type_context(in_c_type_context or isinstance(template_node, CBaseTypeNode)):
                ttype = template_node.analyse_as_type(env)
            if ttype is None:
                if base_type.is_cpp_class:
                    error(template_node.pos, 'unknown type in template argument')
                    ttype = error_type
            elif require_python_types and (not ttype.is_pyobject) or (require_optional_types and (not ttype.can_be_optional())):
                if ttype.equivalent_type and (not template_node.as_cython_attribute()):
                    ttype = ttype.equivalent_type
                else:
                    error(template_node.pos, '%s[...] cannot be applied to type %s' % (base_type.python_type_constructor_name, ttype))
                    ttype = error_type
            template_types.append(ttype)
        return template_types

    def analyse(self, env, could_be_name=False, base_type=None):
        if base_type is None:
            base_type = self.base_type_node.analyse(env)
        if base_type.is_error:
            return base_type
        if base_type.is_cpp_class and base_type.is_template_type() or base_type.python_type_constructor_name:
            if self.keyword_args and self.keyword_args.key_value_pairs:
                tp = 'c++ templates' if base_type.is_cpp_class else 'indexed types'
                error(self.pos, '%s cannot take keyword arguments' % tp)
                self.type = PyrexTypes.error_type
                return self.type
            template_types = self._analyse_template_types(env, base_type)
            self.type = base_type.specialize_here(self.pos, env, template_types)
        elif base_type.is_pyobject:
            from . import Buffer
            options = Buffer.analyse_buffer_options(self.pos, env, self.positional_args, self.keyword_args, base_type.buffer_defaults)
            if sys.version_info[0] < 3:
                options = dict([(name.encode('ASCII'), value) for name, value in options.items()])
            self.type = PyrexTypes.BufferType(base_type, **options)
            if has_np_pythran(env) and is_pythran_buffer(self.type):
                self.type = PyrexTypes.PythranExpr(pythran_type(self.type), self.type)
        else:
            empty_declarator = CNameDeclaratorNode(self.pos, name='', cname=None)
            if len(self.positional_args) > 1 or self.keyword_args.key_value_pairs:
                error(self.pos, 'invalid array declaration')
                self.type = PyrexTypes.error_type
            else:
                if not self.positional_args:
                    dimension = None
                else:
                    dimension = self.positional_args[0]
                self.array_declarator = CArrayDeclaratorNode(self.pos, base=empty_declarator, dimension=dimension)
                self.type = self.array_declarator.analyse(base_type, env)[1]
        if self.type and self.type.is_fused and env.fused_to_specific:
            try:
                self.type = self.type.specialize(env.fused_to_specific)
            except CannotSpecialize:
                error(self.pos, "'%s' cannot be specialized since its type is not a fused argument to this function" % self.name)
        return self.type

    def analyse_pytyping_modifiers(self, env):
        modifiers = []
        modifier_node = self
        while modifier_node.is_templated_type_node and modifier_node.base_type_node and (len(modifier_node.positional_args) == 1):
            modifier_type = self.base_type_node.analyse_as_type(env)
            if modifier_type.python_type_constructor_name and modifier_type.modifier_name:
                modifiers.append(modifier_type.modifier_name)
            modifier_node = modifier_node.positional_args[0]
        return modifiers