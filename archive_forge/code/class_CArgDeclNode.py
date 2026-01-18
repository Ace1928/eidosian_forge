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
class CArgDeclNode(Node):
    child_attrs = ['base_type', 'declarator', 'default', 'annotation']
    outer_attrs = ['default', 'annotation']
    is_self_arg = 0
    is_type_arg = 0
    is_generic = 1
    is_special_method_optional = False
    kw_only = 0
    pos_only = 0
    not_none = 0
    or_none = 0
    type = None
    name_declarator = None
    default_value = None
    annotation = None
    is_dynamic = 0

    def declared_name(self):
        return self.declarator.declared_name()

    @property
    def name_cstring(self):
        return self.name.as_c_string_literal()

    @property
    def hdr_cname(self):
        if self.needs_conversion:
            return punycodify_name(Naming.arg_prefix + self.entry.name)
        else:
            return punycodify_name(Naming.var_prefix + self.entry.name)

    def analyse(self, env, nonempty=0, is_self_arg=False):
        if is_self_arg:
            self.base_type.is_self_arg = self.is_self_arg = is_self_arg
        if self.type is not None:
            return (self.name_declarator, self.type)
        if isinstance(self.declarator, CNameDeclaratorNode) and self.declarator.name == '':
            if nonempty:
                if self.base_type.is_basic_c_type:
                    type = self.base_type.analyse(env, could_be_name=True)
                    arg_name = type.empty_declaration_code()
                else:
                    arg_name = self.base_type.name
                self.declarator.name = EncodedString(arg_name)
                self.base_type.name = None
                self.base_type.is_basic_c_type = False
            could_be_name = True
        else:
            could_be_name = False
        self.base_type.is_arg = True
        base_type = self.base_type.analyse(env, could_be_name=could_be_name)
        base_arg_name = getattr(self.base_type, 'arg_name', None)
        if base_arg_name:
            self.declarator.name = base_arg_name
        if base_type.is_array and isinstance(self.base_type, TemplatedTypeNode) and isinstance(self.declarator, CArrayDeclaratorNode):
            declarator = self.declarator
            while isinstance(declarator.base, CArrayDeclaratorNode):
                declarator = declarator.base
            declarator.base = self.base_type.array_declarator
            base_type = base_type.base_type
        if self.annotation and env and env.directives['annotation_typing'] and (getattr(self.base_type, 'name', None) is None):
            arg_type = self.inject_type_from_annotations(env)
            if arg_type is not None:
                base_type = arg_type
        return self.declarator.analyse(base_type, env, nonempty=nonempty)

    def inject_type_from_annotations(self, env):
        annotation = self.annotation
        if not annotation:
            return None
        modifiers, arg_type = annotation.analyse_type_annotation(env, assigned_value=self.default)
        if arg_type is not None:
            self.base_type = CAnalysedBaseTypeNode(annotation.pos, type=arg_type, is_arg=True)
        if arg_type:
            if 'typing.Optional' in modifiers:
                arg_type = arg_type.resolve()
                if arg_type and (not arg_type.can_be_optional()):
                    pass
                else:
                    self.or_none = True
            elif arg_type is py_object_type:
                self.or_none = True
            elif self.default and self.default.is_none and (arg_type.can_be_optional() or arg_type.equivalent_type):
                if not arg_type.can_be_optional():
                    arg_type = arg_type.equivalent_type
                if not self.or_none:
                    warning(self.pos, "PEP-484 recommends 'typing.Optional[...]' for arguments that can be None.")
                    self.or_none = True
            elif not self.or_none and arg_type.can_be_optional():
                self.not_none = True
        return arg_type

    def calculate_default_value_code(self, code):
        if self.default_value is None:
            if self.default:
                if self.default.is_literal:
                    self.default.generate_evaluation_code(code)
                    return self.type.cast_code(self.default.result())
                self.default_value = code.get_argument_default_const(self.type)
        return self.default_value

    def annotate(self, code):
        if self.default:
            self.default.annotate(code)

    def generate_assignment_code(self, code, target=None, overloaded_assignment=False):
        default = self.default
        if default is None or default.is_literal:
            return
        if target is None:
            target = self.calculate_default_value_code(code)
        default.generate_evaluation_code(code)
        default.make_owned_reference(code)
        result = default.result() if overloaded_assignment else default.result_as(self.type)
        code.putln('%s = %s;' % (target, result))
        code.put_giveref(default.result(), self.type)
        default.generate_post_assignment_code(code)
        default.free_temps(code)