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
class CppClassNode(CStructOrUnionDefNode, BlockNode):
    decorators = None

    def declare(self, env):
        if self.templates is None:
            template_types = None
        else:
            template_types = [PyrexTypes.TemplatePlaceholderType(template_name, not required) for template_name, required in self.templates]
            num_optional_templates = sum((not required for _, required in self.templates))
            if num_optional_templates and (not all((required for _, required in self.templates[:-num_optional_templates]))):
                error(self.pos, 'Required template parameters must precede optional template parameters.')
        self.entry = env.declare_cpp_class(self.name, None, self.pos, self.cname, base_classes=[], visibility=self.visibility, templates=template_types)

    def analyse_declarations(self, env):
        if self.templates is None:
            template_types = template_names = None
        else:
            template_names = [template_name for template_name, _ in self.templates]
            template_types = [PyrexTypes.TemplatePlaceholderType(template_name, not required) for template_name, required in self.templates]
        scope = None
        if self.attributes is not None:
            scope = CppClassScope(self.name, env, templates=template_names)

        def base_ok(base_class):
            if base_class.is_cpp_class or base_class.is_struct:
                return True
            else:
                error(self.pos, "Base class '%s' not a struct or class." % base_class)
        base_class_types = filter(base_ok, [b.analyse(scope or env) for b in self.base_classes])
        self.entry = env.declare_cpp_class(self.name, scope, self.pos, self.cname, base_class_types, visibility=self.visibility, templates=template_types)
        if self.entry is None:
            return
        self.entry.is_cpp_class = 1
        if scope is not None:
            scope.type = self.entry.type
        defined_funcs = []

        def func_attributes(attributes):
            for attr in attributes:
                if isinstance(attr, CFuncDefNode):
                    yield attr
                elif isinstance(attr, CompilerDirectivesNode):
                    for sub_attr in func_attributes(attr.body.stats):
                        yield sub_attr
                elif isinstance(attr, CppClassNode) and attr.attributes is not None:
                    for sub_attr in func_attributes(attr.attributes):
                        yield sub_attr
        if self.attributes is not None:
            if self.in_pxd and (not env.in_cinclude):
                self.entry.defined_in_pxd = 1
            for attr in self.attributes:
                declare = getattr(attr, 'declare', None)
                if declare:
                    attr.declare(scope)
                attr.analyse_declarations(scope)
            for func in func_attributes(self.attributes):
                defined_funcs.append(func)
                if self.templates is not None:
                    func.template_declaration = 'template <typename %s>' % ', typename '.join(template_names)
        self.body = StatListNode(self.pos, stats=defined_funcs)
        self.scope = scope

    def analyse_expressions(self, env):
        self.body = self.body.analyse_expressions(self.entry.type.scope)
        return self

    def generate_function_definitions(self, env, code):
        self.body.generate_function_definitions(self.entry.type.scope, code)

    def generate_execution_code(self, code):
        self.body.generate_execution_code(code)

    def annotate(self, code):
        self.body.annotate(code)