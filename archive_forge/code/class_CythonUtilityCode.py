from __future__ import absolute_import
from .TreeFragment import parse_from_strings, StringParseContext
from . import Symtab
from . import Naming
from . import Code
class CythonUtilityCode(Code.UtilityCodeBase):
    """
    Utility code written in the Cython language itself.

    The @cname decorator can set the cname for a function, method of cdef class.
    Functions decorated with @cname('c_func_name') get the given cname.

    For cdef classes the rules are as follows:
        obj struct      -> <cname>_obj
        obj type ptr    -> <cname>_type
        methods         -> <class_cname>_<method_cname>

    For methods the cname decorator is optional, but without the decorator the
    methods will not be prototyped. See Cython.Compiler.CythonScope and
    tests/run/cythonscope.pyx for examples.
    """
    is_cython_utility = True

    def __init__(self, impl, name='__pyxutil', prefix='', requires=None, file=None, from_scope=None, context=None, compiler_directives=None, outer_module_scope=None):
        context_types = {}
        if context is not None:
            from .PyrexTypes import BaseType
            for key, value in context.items():
                if isinstance(value, BaseType):
                    context[key] = key
                    context_types[key] = value
            impl = Code.sub_tempita(impl, context, file, name)
        self.impl = impl
        self.name = name
        self.file = file
        self.prefix = prefix
        self.requires = requires or []
        self.from_scope = from_scope
        self.outer_module_scope = outer_module_scope
        self.compiler_directives = compiler_directives
        self.context_types = context_types

    def __eq__(self, other):
        if isinstance(other, CythonUtilityCode):
            return self._equality_params() == other._equality_params()
        else:
            return False

    def _equality_params(self):
        outer_scope = self.outer_module_scope
        while isinstance(outer_scope, NonManglingModuleScope):
            outer_scope = outer_scope.outer_scope
        return (self.impl, outer_scope, self.compiler_directives)

    def __hash__(self):
        return hash(self.impl)

    def get_tree(self, entries_only=False, cython_scope=None):
        from .AnalysedTreeTransforms import AutoTestDictTransform
        excludes = [AutoTestDictTransform]
        from . import Pipeline, ParseTreeTransforms
        context = CythonUtilityCodeContext(self.name, compiler_directives=self.compiler_directives, cpp=cython_scope.is_cpp() if cython_scope else False)
        context.prefix = self.prefix
        context.cython_scope = cython_scope
        tree = parse_from_strings(self.name, self.impl, context=context, allow_struct_enum_decorator=True, in_utility_code=True)
        pipeline = Pipeline.create_pipeline(context, 'pyx', exclude_classes=excludes)
        if entries_only:
            p = []
            for t in pipeline:
                p.append(t)
                if isinstance(t, ParseTreeTransforms.AnalyseDeclarationsTransform):
                    break
            pipeline = p
        transform = ParseTreeTransforms.CnameDirectivesTransform(context)
        before = ParseTreeTransforms.InterpretCompilerDirectives
        pipeline = Pipeline.insert_into_pipeline(pipeline, transform, before=before)

        def merge_scope(scope):

            def merge_scope_transform(module_node):
                module_node.scope.merge_in(scope)
                return module_node
            return merge_scope_transform
        if self.from_scope:
            pipeline = Pipeline.insert_into_pipeline(pipeline, merge_scope(self.from_scope), before=ParseTreeTransforms.AnalyseDeclarationsTransform)
        for dep in self.requires:
            if isinstance(dep, CythonUtilityCode) and hasattr(dep, 'tree') and (not cython_scope):
                pipeline = Pipeline.insert_into_pipeline(pipeline, merge_scope(dep.tree.scope), before=ParseTreeTransforms.AnalyseDeclarationsTransform)
        if self.outer_module_scope:

            def scope_transform(module_node):
                module_node.scope.outer_scope = self.outer_module_scope
                return module_node
            pipeline = Pipeline.insert_into_pipeline(pipeline, scope_transform, before=ParseTreeTransforms.AnalyseDeclarationsTransform)
        if self.context_types:

            def scope_transform(module_node):
                dummy_entry = object()
                for name, type in self.context_types.items():
                    old_type_entry = getattr(type, 'entry', dummy_entry)
                    entry = module_node.scope.declare_type(name, type, None, visibility='extern')
                    if old_type_entry is not dummy_entry:
                        type.entry = old_type_entry
                    entry.in_cinclude = True
                return module_node
            pipeline = Pipeline.insert_into_pipeline(pipeline, scope_transform, before=ParseTreeTransforms.AnalyseDeclarationsTransform)
        err, tree = Pipeline.run_pipeline(pipeline, tree, printtree=False)
        assert not err, err
        self.tree = tree
        return tree

    def put_code(self, output):
        pass

    @classmethod
    def load_as_string(cls, util_code_name, from_file=None, **kwargs):
        """
        Load a utility code as a string. Returns (proto, implementation)
        """
        util = cls.load(util_code_name, from_file, **kwargs)
        return (util.proto, util.impl)

    def declare_in_scope(self, dest_scope, used=False, cython_scope=None, allowlist=None):
        """
        Declare all entries from the utility code in dest_scope. Code will only
        be included for used entries. If module_name is given, declare the
        type entries with that name.
        """
        tree = self.get_tree(entries_only=True, cython_scope=cython_scope)
        entries = tree.scope.entries
        entries.pop('__name__')
        entries.pop('__file__')
        entries.pop('__builtins__')
        entries.pop('__doc__')
        for entry in entries.values():
            entry.utility_code_definition = self
            entry.used = used
        original_scope = tree.scope
        dest_scope.merge_in(original_scope, merge_unused=True, allowlist=allowlist)
        tree.scope = dest_scope
        for dep in self.requires:
            if dep.is_cython_utility:
                dep.declare_in_scope(dest_scope, cython_scope=cython_scope)
        return original_scope

    @staticmethod
    def filter_inherited_directives(current_directives):
        """
        Cython utility code should usually only pick up a few directives from the
        environment (those that intentionally control its function) and ignore most
        other compiler directives. This function provides a sensible default list
        of directives to copy.
        """
        from .Options import _directive_defaults
        utility_code_directives = dict(_directive_defaults)
        inherited_directive_names = ('binding', 'always_allow_keywords', 'allow_none_for_extension_args', 'auto_pickle', 'ccomplex', 'c_string_type', 'c_string_encoding', 'optimize.inline_defnode_calls', 'optimize.unpack_method_calls', 'optimize.unpack_method_calls_in_pyinit', 'optimize.use_switch')
        for name in inherited_directive_names:
            if name in current_directives:
                utility_code_directives[name] = current_directives[name]
        return utility_code_directives