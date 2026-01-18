from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
class InterpretCompilerDirectives(CythonTransform):
    """
    After parsing, directives can be stored in a number of places:
    - #cython-comments at the top of the file (stored in ModuleNode)
    - Command-line arguments overriding these
    - @cython.directivename decorators
    - with cython.directivename: statements
    - replaces "cython.compiled" with BoolNode(value=True)
      allowing unreachable blocks to be removed at a fairly early stage
      before cython typing rules are forced on applied

    This transform is responsible for interpreting these various sources
    and store the directive in two ways:
    - Set the directives attribute of the ModuleNode for global directives.
    - Use a CompilerDirectivesNode to override directives for a subtree.

    (The first one is primarily to not have to modify with the tree
    structure, so that ModuleNode stay on top.)

    The directives are stored in dictionaries from name to value in effect.
    Each such dictionary is always filled in for all possible directives,
    using default values where no value is given by the user.

    The available directives are controlled in Options.py.

    Note that we have to run this prior to analysis, and so some minor
    duplication of functionality has to occur: We manually track cimports
    and which names the "cython" module may have been imported to.
    """
    unop_method_nodes = {'typeof': ExprNodes.TypeofNode, 'operator.address': ExprNodes.AmpersandNode, 'operator.dereference': ExprNodes.DereferenceNode, 'operator.preincrement': ExprNodes.inc_dec_constructor(True, '++'), 'operator.predecrement': ExprNodes.inc_dec_constructor(True, '--'), 'operator.postincrement': ExprNodes.inc_dec_constructor(False, '++'), 'operator.postdecrement': ExprNodes.inc_dec_constructor(False, '--'), 'operator.typeid': ExprNodes.TypeidNode, 'address': ExprNodes.AmpersandNode}
    binop_method_nodes = {'operator.comma': ExprNodes.c_binop_constructor(',')}
    special_methods = {'declare', 'union', 'struct', 'typedef', 'sizeof', 'cast', 'pointer', 'compiled', 'NULL', 'fused_type', 'parallel'}
    special_methods.update(unop_method_nodes)
    valid_cython_submodules = {'cimports', 'dataclasses', 'operator', 'parallel', 'view'}
    valid_parallel_directives = {'parallel', 'prange', 'threadid'}

    def __init__(self, context, compilation_directive_defaults):
        super(InterpretCompilerDirectives, self).__init__(context)
        self.cython_module_names = set()
        self.directive_names = {'staticmethod': 'staticmethod'}
        self.parallel_directives = {}
        directives = copy.deepcopy(Options.get_directive_defaults())
        for key, value in compilation_directive_defaults.items():
            directives[_unicode(key)] = copy.deepcopy(value)
        self.directives = directives

    def check_directive_scope(self, pos, directive, scope):
        legal_scopes = Options.directive_scopes.get(directive, None)
        if legal_scopes and scope not in legal_scopes:
            self.context.nonfatal_error(PostParseError(pos, 'The %s compiler directive is not allowed in %s scope' % (directive, scope)))
            return False
        else:
            if directive not in Options.directive_types:
                error(pos, "Invalid directive: '%s'." % (directive,))
            return True

    def _check_valid_cython_module(self, pos, module_name):
        if not module_name.startswith('cython.'):
            return
        submodule = module_name.split('.', 2)[1]
        if submodule in self.valid_cython_submodules:
            return
        extra = ''
        hints = [line.split() for line in '                imp                  cimports\n                cimp                 cimports\n                para                 parallel\n                parra                parallel\n                dataclass            dataclasses\n            '.splitlines()[:-1]]
        for wrong, correct in hints:
            if module_name.startswith('cython.' + wrong):
                extra = "Did you mean 'cython.%s' ?" % correct
                break
        if not extra:
            is_simple_cython_name = submodule in Options.directive_types
            if not is_simple_cython_name and (not submodule.startswith('_')):
                from .. import Shadow
                is_simple_cython_name = hasattr(Shadow, submodule)
            if is_simple_cython_name:
                extra = "Instead, use 'import cython' and then 'cython.%s'." % submodule
        error(pos, "'%s' is not a valid cython.* module%s%s" % (module_name, '. ' if extra else '', extra))

    def visit_ModuleNode(self, node):
        for key in sorted(node.directive_comments):
            if not self.check_directive_scope(node.pos, key, 'module'):
                self.wrong_scope_error(node.pos, key, 'module')
                del node.directive_comments[key]
        self.module_scope = node.scope
        self.directives.update(node.directive_comments)
        node.directives = self.directives
        node.parallel_directives = self.parallel_directives
        self.visitchildren(node)
        node.cython_module_names = self.cython_module_names
        return node

    def visit_CompilerDirectivesNode(self, node):
        old_directives, self.directives = (self.directives, node.directives)
        self.visitchildren(node)
        self.directives = old_directives
        return node

    def is_cython_directive(self, name):
        return name in Options.directive_types or name in self.special_methods or PyrexTypes.parse_basic_type(name)

    def is_parallel_directive(self, full_name, pos):
        """
        Checks to see if fullname (e.g. cython.parallel.prange) is a valid
        parallel directive. If it is a star import it also updates the
        parallel_directives.
        """
        result = (full_name + '.').startswith('cython.parallel.')
        if result:
            directive = full_name.split('.')
            if full_name == u'cython.parallel':
                self.parallel_directives[u'parallel'] = u'cython.parallel'
            elif full_name == u'cython.parallel.*':
                for name in self.valid_parallel_directives:
                    self.parallel_directives[name] = u'cython.parallel.%s' % name
            elif len(directive) != 3 or directive[-1] not in self.valid_parallel_directives:
                error(pos, 'No such directive: %s' % full_name)
            self.module_scope.use_utility_code(UtilityCode.load_cached('InitThreads', 'ModuleSetupCode.c'))
        return result

    def visit_CImportStatNode(self, node):
        module_name = node.module_name
        if module_name == u'cython.cimports':
            error(node.pos, "Cannot cimport the 'cython.cimports' package directly, only submodules.")
        if module_name.startswith(u'cython.cimports.'):
            if node.as_name and node.as_name != u'cython':
                node.module_name = module_name[len(u'cython.cimports.'):]
                return node
            error(node.pos, "Python cimports must use 'from cython.cimports... import ...' or 'import ... as ...', not just 'import ...'")
        if module_name == u'cython':
            self.cython_module_names.add(node.as_name or u'cython')
        elif module_name.startswith(u'cython.'):
            if module_name.startswith(u'cython.parallel.'):
                error(node.pos, node.module_name + ' is not a module')
            else:
                self._check_valid_cython_module(node.pos, module_name)
            if module_name == u'cython.parallel':
                if node.as_name and node.as_name != u'cython':
                    self.parallel_directives[node.as_name] = module_name
                else:
                    self.cython_module_names.add(u'cython')
                    self.parallel_directives[u'cython.parallel'] = module_name
                self.module_scope.use_utility_code(UtilityCode.load_cached('InitThreads', 'ModuleSetupCode.c'))
            elif node.as_name:
                self.directive_names[node.as_name] = module_name[7:]
            else:
                self.cython_module_names.add(u'cython')
            return None
        return node

    def visit_FromCImportStatNode(self, node):
        module_name = node.module_name
        if module_name == u'cython.cimports' or module_name.startswith(u'cython.cimports.'):
            return self._create_cimport_from_import(node.pos, module_name, node.relative_level, node.imported_names)
        elif not node.relative_level and (module_name == u'cython' or module_name.startswith(u'cython.')):
            self._check_valid_cython_module(node.pos, module_name)
            submodule = (module_name + u'.')[7:]
            newimp = []
            for pos, name, as_name in node.imported_names:
                full_name = submodule + name
                qualified_name = u'cython.' + full_name
                if self.is_parallel_directive(qualified_name, node.pos):
                    self.parallel_directives[as_name or name] = qualified_name
                elif self.is_cython_directive(full_name):
                    self.directive_names[as_name or name] = full_name
                elif full_name in ['dataclasses', 'typing']:
                    self.directive_names[as_name or name] = full_name
                    newimp.append((pos, name, as_name))
                else:
                    newimp.append((pos, name, as_name))
            if not newimp:
                return None
            node.imported_names = newimp
        return node

    def visit_FromImportStatNode(self, node):
        import_node = node.module
        module_name = import_node.module_name.value
        if module_name == u'cython.cimports' or module_name.startswith(u'cython.cimports.'):
            imported_names = []
            for name, name_node in node.items:
                imported_names.append((name_node.pos, name, None if name == name_node.name else name_node.name))
            return self._create_cimport_from_import(node.pos, module_name, import_node.level, imported_names)
        elif module_name == u'cython' or module_name.startswith(u'cython.'):
            self._check_valid_cython_module(import_node.module_name.pos, module_name)
            submodule = (module_name + u'.')[7:]
            newimp = []
            for name, name_node in node.items:
                full_name = submodule + name
                qualified_name = u'cython.' + full_name
                if self.is_parallel_directive(qualified_name, node.pos):
                    self.parallel_directives[name_node.name] = qualified_name
                elif self.is_cython_directive(full_name):
                    self.directive_names[name_node.name] = full_name
                else:
                    newimp.append((name, name_node))
            if not newimp:
                return None
            node.items = newimp
        return node

    def _create_cimport_from_import(self, node_pos, module_name, level, imported_names):
        if module_name == u'cython.cimports' or module_name.startswith(u'cython.cimports.'):
            module_name = EncodedString(module_name[len(u'cython.cimports.'):])
        if module_name:
            return Nodes.FromCImportStatNode(node_pos, module_name=module_name, relative_level=level, imported_names=imported_names)
        else:
            return [Nodes.CImportStatNode(pos, module_name=dotted_name, as_name=as_name, is_absolute=level == 0) for pos, dotted_name, as_name in imported_names]

    def visit_SingleAssignmentNode(self, node):
        if isinstance(node.rhs, ExprNodes.ImportNode):
            module_name = node.rhs.module_name.value
            if module_name != u'cython' and (not module_name.startswith('cython.')):
                return node
            node = Nodes.CImportStatNode(node.pos, module_name=module_name, as_name=node.lhs.name)
            node = self.visit_CImportStatNode(node)
        else:
            self.visitchildren(node)
        return node

    def visit_NameNode(self, node):
        if node.annotation:
            self.visitchild(node, 'annotation')
        if node.name in self.cython_module_names:
            node.is_cython_module = True
        else:
            directive = self.directive_names.get(node.name)
            if directive is not None:
                node.cython_attribute = directive
        if node.as_cython_attribute() == 'compiled':
            return ExprNodes.BoolNode(node.pos, value=True)
        return node

    def visit_AttributeNode(self, node):
        self.visitchildren(node)
        if node.as_cython_attribute() == 'compiled':
            return ExprNodes.BoolNode(node.pos, value=True)
        return node

    def visit_AnnotationNode(self, node):
        if node.expr:
            self.visit(node.expr)
        return node

    def visit_NewExprNode(self, node):
        self.visitchild(node, 'cppclass')
        self.visitchildren(node)
        return node

    def try_to_parse_directives(self, node):
        if isinstance(node, ExprNodes.CallNode):
            self.visitchild(node, 'function')
            optname = node.function.as_cython_attribute()
            if optname:
                directivetype = Options.directive_types.get(optname)
                if directivetype:
                    args, kwds = node.explicit_args_kwds()
                    directives = []
                    key_value_pairs = []
                    if kwds is not None and directivetype is not dict:
                        for keyvalue in kwds.key_value_pairs:
                            key, value = keyvalue
                            sub_optname = '%s.%s' % (optname, key.value)
                            if Options.directive_types.get(sub_optname):
                                directives.append(self.try_to_parse_directive(sub_optname, [value], None, keyvalue.pos))
                            else:
                                key_value_pairs.append(keyvalue)
                        if not key_value_pairs:
                            kwds = None
                        else:
                            kwds.key_value_pairs = key_value_pairs
                        if directives and (not kwds) and (not args):
                            return directives
                    directives.append(self.try_to_parse_directive(optname, args, kwds, node.function.pos))
                    return directives
        elif isinstance(node, (ExprNodes.AttributeNode, ExprNodes.NameNode)):
            self.visit(node)
            optname = node.as_cython_attribute()
            if optname:
                directivetype = Options.directive_types.get(optname)
                if directivetype is bool:
                    arg = ExprNodes.BoolNode(node.pos, value=True)
                    return [self.try_to_parse_directive(optname, [arg], None, node.pos)]
                elif directivetype is None or directivetype is Options.DEFER_ANALYSIS_OF_ARGUMENTS:
                    return [(optname, None)]
                else:
                    raise PostParseError(node.pos, "The '%s' directive should be used as a function call." % optname)
        return None

    def try_to_parse_directive(self, optname, args, kwds, pos):
        if optname == 'np_pythran' and (not self.context.cpp):
            raise PostParseError(pos, 'The %s directive can only be used in C++ mode.' % optname)
        elif optname == 'exceptval':
            arg_error = len(args) > 1
            check = True
            if kwds and kwds.key_value_pairs:
                kw = kwds.key_value_pairs[0]
                if len(kwds.key_value_pairs) == 1 and kw.key.is_string_literal and (kw.key.value == 'check') and isinstance(kw.value, ExprNodes.BoolNode):
                    check = kw.value.value
                else:
                    arg_error = True
            if arg_error:
                raise PostParseError(pos, 'The exceptval directive takes 0 or 1 positional arguments and the boolean keyword "check"')
            return ('exceptval', (args[0] if args else None, check))
        directivetype = Options.directive_types.get(optname)
        if len(args) == 1 and isinstance(args[0], ExprNodes.NoneNode):
            return (optname, Options.get_directive_defaults()[optname])
        elif directivetype is bool:
            if kwds is not None or len(args) != 1 or (not isinstance(args[0], ExprNodes.BoolNode)):
                raise PostParseError(pos, 'The %s directive takes one compile-time boolean argument' % optname)
            return (optname, args[0].value)
        elif directivetype is int:
            if kwds is not None or len(args) != 1 or (not isinstance(args[0], ExprNodes.IntNode)):
                raise PostParseError(pos, 'The %s directive takes one compile-time integer argument' % optname)
            return (optname, int(args[0].value))
        elif directivetype is str:
            if kwds is not None or len(args) != 1 or (not isinstance(args[0], (ExprNodes.StringNode, ExprNodes.UnicodeNode))):
                raise PostParseError(pos, 'The %s directive takes one compile-time string argument' % optname)
            return (optname, str(args[0].value))
        elif directivetype is type:
            if kwds is not None or len(args) != 1:
                raise PostParseError(pos, 'The %s directive takes one type argument' % optname)
            return (optname, args[0])
        elif directivetype is dict:
            if len(args) != 0:
                raise PostParseError(pos, 'The %s directive takes no prepositional arguments' % optname)
            return (optname, kwds.as_python_dict())
        elif directivetype is list:
            if kwds and len(kwds.key_value_pairs) != 0:
                raise PostParseError(pos, 'The %s directive takes no keyword arguments' % optname)
            return (optname, [str(arg.value) for arg in args])
        elif callable(directivetype):
            if kwds is not None or len(args) != 1 or (not isinstance(args[0], (ExprNodes.StringNode, ExprNodes.UnicodeNode))):
                raise PostParseError(pos, 'The %s directive takes one compile-time string argument' % optname)
            return (optname, directivetype(optname, str(args[0].value)))
        elif directivetype is Options.DEFER_ANALYSIS_OF_ARGUMENTS:
            return (optname, (args, kwds.as_python_dict() if kwds else {}))
        else:
            assert False

    def visit_with_directives(self, node, directives, contents_directives):
        if not directives:
            assert not contents_directives
            return self.visit_Node(node)
        old_directives = self.directives
        new_directives = Options.copy_inherited_directives(old_directives, **directives)
        if contents_directives is not None:
            new_contents_directives = Options.copy_inherited_directives(old_directives, **contents_directives)
        else:
            new_contents_directives = new_directives
        if new_directives == old_directives:
            return self.visit_Node(node)
        self.directives = new_directives
        if contents_directives is not None and new_contents_directives != new_directives:
            node.body = Nodes.StatListNode(node.body.pos, stats=[Nodes.CompilerDirectivesNode(node.body.pos, directives=new_contents_directives, body=node.body)])
        retbody = self.visit_Node(node)
        self.directives = old_directives
        if not isinstance(retbody, Nodes.StatListNode):
            retbody = Nodes.StatListNode(node.pos, stats=[retbody])
        return Nodes.CompilerDirectivesNode(pos=retbody.pos, body=retbody, directives=new_directives)

    def visit_FuncDefNode(self, node):
        directives, contents_directives = self._extract_directives(node, 'function')
        return self.visit_with_directives(node, directives, contents_directives)

    def visit_CVarDefNode(self, node):
        directives, _ = self._extract_directives(node, 'function')
        for name, value in directives.items():
            if name == 'locals':
                node.directive_locals = value
            elif name not in ('final', 'staticmethod'):
                self.context.nonfatal_error(PostParseError(node.pos, 'Cdef functions can only take cython.locals(), staticmethod, or final decorators, got %s.' % name))
        return self.visit_with_directives(node, directives, contents_directives=None)

    def visit_CClassDefNode(self, node):
        directives, contents_directives = self._extract_directives(node, 'cclass')
        return self.visit_with_directives(node, directives, contents_directives)

    def visit_CppClassNode(self, node):
        directives, contents_directives = self._extract_directives(node, 'cppclass')
        return self.visit_with_directives(node, directives, contents_directives)

    def visit_PyClassDefNode(self, node):
        directives, contents_directives = self._extract_directives(node, 'class')
        return self.visit_with_directives(node, directives, contents_directives)

    def _extract_directives(self, node, scope_name):
        """
        Returns two dicts - directives applied to this function/class
        and directives applied to its contents. They aren't always the
        same (since e.g. cfunc should not be applied to inner functions)
        """
        if not node.decorators:
            return ({}, {})
        directives = []
        realdecs = []
        both = []
        current_opt_dict = dict(self.directives)
        missing = object()
        for dec in node.decorators[::-1]:
            new_directives = self.try_to_parse_directives(dec.decorator)
            if new_directives is not None:
                for directive in new_directives:
                    if self.check_directive_scope(node.pos, directive[0], scope_name):
                        name, value = directive
                        if name in ('nogil', 'with_gil'):
                            if value is None:
                                value = True
                            else:
                                args, kwds = value
                                if kwds or len(args) != 1 or (not isinstance(args[0], ExprNodes.BoolNode)):
                                    raise PostParseError(dec.pos, 'The %s directive takes one compile-time boolean argument' % name)
                                value = args[0].value
                            directive = (name, value)
                        if current_opt_dict.get(name, missing) != value:
                            if name == 'cfunc' and 'ufunc' in current_opt_dict:
                                error(dec.pos, 'Cannot apply @cfunc to @ufunc, please reverse the decorators.')
                            directives.append(directive)
                            current_opt_dict[name] = value
                        else:
                            warning(dec.pos, 'Directive does not change previous value (%s%s)' % (name, '=%r' % value if value is not None else ''))
                        if directive[0] == 'staticmethod':
                            both.append(dec)
                    if directive[0] == 'cclass' and scope_name == 'class':
                        scope_name = 'cclass'
            else:
                realdecs.append(dec)
        node.decorators = realdecs[::-1] + both[::-1]
        optdict = {}
        contents_optdict = {}
        for name, value in directives:
            if name in optdict:
                old_value = optdict[name]
                if isinstance(old_value, dict):
                    old_value.update(value)
                elif isinstance(old_value, list):
                    old_value.extend(value)
                else:
                    optdict[name] = value
            else:
                optdict[name] = value
            if name not in Options.immediate_decorator_directives:
                contents_optdict[name] = value
        return (optdict, contents_optdict)

    def visit_WithStatNode(self, node):
        directive_dict = {}
        for directive in self.try_to_parse_directives(node.manager) or []:
            if directive is not None:
                if node.target is not None:
                    self.context.nonfatal_error(PostParseError(node.pos, "Compiler directive with statements cannot contain 'as'"))
                else:
                    name, value = directive
                    if name in ('nogil', 'gil'):
                        condition = None
                        if isinstance(node.manager, ExprNodes.SimpleCallNode) and len(node.manager.args) > 0:
                            if len(node.manager.args) == 1:
                                condition = node.manager.args[0]
                            else:
                                self.context.nonfatal_error(PostParseError(node.pos, 'Compiler directive %s accepts one positional argument.' % name))
                        elif isinstance(node.manager, ExprNodes.GeneralCallNode):
                            self.context.nonfatal_error(PostParseError(node.pos, 'Compiler directive %s accepts one positional argument.' % name))
                        node = Nodes.GILStatNode(node.pos, state=name, body=node.body, condition=condition)
                        return self.visit_Node(node)
                    if self.check_directive_scope(node.pos, name, 'with statement'):
                        directive_dict[name] = value
        if directive_dict:
            return self.visit_with_directives(node.body, directive_dict, contents_directives=None)
        return self.visit_Node(node)