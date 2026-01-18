from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
class DeclarationWriter(TreeVisitor):
    """
    A Cython code writer that is limited to declarations nodes.
    """
    indent_string = u'    '

    def __init__(self, result=None):
        super(DeclarationWriter, self).__init__()
        if result is None:
            result = LinesResult()
        self.result = result
        self.numindents = 0
        self.tempnames = {}
        self.tempblockindex = 0

    def write(self, tree):
        self.visit(tree)
        return self.result

    def indent(self):
        self.numindents += 1

    def dedent(self):
        self.numindents -= 1

    def startline(self, s=u''):
        self.result.put(self.indent_string * self.numindents + s)

    def put(self, s):
        self.result.put(s)

    def putline(self, s):
        self.result.putline(self.indent_string * self.numindents + s)

    def endline(self, s=u''):
        self.result.putline(s)

    def line(self, s):
        self.startline(s)
        self.endline()

    def comma_separated_list(self, items, output_rhs=False):
        if len(items) > 0:
            for item in items[:-1]:
                self.visit(item)
                if output_rhs and item.default is not None:
                    self.put(u' = ')
                    self.visit(item.default)
                self.put(u', ')
            self.visit(items[-1])
            if output_rhs and items[-1].default is not None:
                self.put(u' = ')
                self.visit(items[-1].default)

    def _visit_indented(self, node):
        self.indent()
        self.visit(node)
        self.dedent()

    def visit_Node(self, node):
        raise AssertionError('Node not handled by serializer: %r' % node)

    def visit_ModuleNode(self, node):
        self.visitchildren(node)

    def visit_StatListNode(self, node):
        self.visitchildren(node)

    def visit_CDefExternNode(self, node):
        if node.include_file is None:
            file = u'*'
        else:
            file = u'"%s"' % node.include_file
        self.putline(u'cdef extern from %s:' % file)
        self._visit_indented(node.body)

    def visit_CPtrDeclaratorNode(self, node):
        self.put('*')
        self.visit(node.base)

    def visit_CReferenceDeclaratorNode(self, node):
        self.put('&')
        self.visit(node.base)

    def visit_CArrayDeclaratorNode(self, node):
        self.visit(node.base)
        self.put(u'[')
        if node.dimension is not None:
            self.visit(node.dimension)
        self.put(u']')

    def visit_CFuncDeclaratorNode(self, node):
        self.visit(node.base)
        self.put(u'(')
        self.comma_separated_list(node.args)
        self.endline(u')')

    def visit_CNameDeclaratorNode(self, node):
        self.put(node.name)

    def visit_CSimpleBaseTypeNode(self, node):
        if node.is_basic_c_type:
            self.put(('unsigned ', '', 'signed ')[node.signed])
            if node.longness < 0:
                self.put('short ' * -node.longness)
            elif node.longness > 0:
                self.put('long ' * node.longness)
        if node.name is not None:
            self.put(node.name)

    def visit_CComplexBaseTypeNode(self, node):
        self.visit(node.base_type)
        self.visit(node.declarator)

    def visit_CNestedBaseTypeNode(self, node):
        self.visit(node.base_type)
        self.put(u'.')
        self.put(node.name)

    def visit_TemplatedTypeNode(self, node):
        self.visit(node.base_type_node)
        self.put(u'[')
        self.comma_separated_list(node.positional_args + node.keyword_args.key_value_pairs)
        self.put(u']')

    def visit_CVarDefNode(self, node):
        self.startline(u'cdef ')
        self.visit(node.base_type)
        self.put(u' ')
        self.comma_separated_list(node.declarators, output_rhs=True)
        self.endline()

    def _visit_container_node(self, node, decl, extras, attributes):
        self.startline(decl)
        if node.name:
            self.put(u' ')
            self.put(node.name)
            if node.cname is not None:
                self.put(u' "%s"' % node.cname)
        if extras:
            self.put(extras)
        self.endline(':')
        self.indent()
        if not attributes:
            self.putline('pass')
        else:
            for attribute in attributes:
                self.visit(attribute)
        self.dedent()

    def visit_CStructOrUnionDefNode(self, node):
        if node.typedef_flag:
            decl = u'ctypedef '
        else:
            decl = u'cdef '
        if node.visibility == 'public':
            decl += u'public '
        if node.packed:
            decl += u'packed '
        decl += node.kind
        self._visit_container_node(node, decl, None, node.attributes)

    def visit_CppClassNode(self, node):
        extras = ''
        if node.templates:
            extras = u'[%s]' % ', '.join(node.templates)
        if node.base_classes:
            extras += '(%s)' % ', '.join(node.base_classes)
        self._visit_container_node(node, u'cdef cppclass', extras, node.attributes)

    def visit_CEnumDefNode(self, node):
        self._visit_container_node(node, u'cdef enum', None, node.items)

    def visit_CEnumDefItemNode(self, node):
        self.startline(node.name)
        if node.cname:
            self.put(u' "%s"' % node.cname)
        if node.value:
            self.put(u' = ')
            self.visit(node.value)
        self.endline()

    def visit_CClassDefNode(self, node):
        assert not node.module_name
        if node.decorators:
            for decorator in node.decorators:
                self.visit(decorator)
        self.startline(u'cdef class ')
        self.put(node.class_name)
        if node.base_class_name:
            self.put(u'(')
            if node.base_class_module:
                self.put(node.base_class_module)
                self.put(u'.')
            self.put(node.base_class_name)
            self.put(u')')
        self.endline(u':')
        self._visit_indented(node.body)

    def visit_CTypeDefNode(self, node):
        self.startline(u'ctypedef ')
        self.visit(node.base_type)
        self.put(u' ')
        self.visit(node.declarator)
        self.endline()

    def visit_FuncDefNode(self, node):
        self.startline(u'def %s(' % node.name)
        self.comma_separated_list(node.args)
        self.endline(u'):')
        self._visit_indented(node.body)

    def visit_CFuncDefNode(self, node):
        self.startline(u'cpdef ' if node.overridable else u'cdef ')
        if node.modifiers:
            self.put(' '.join(node.modifiers))
            self.put(' ')
        if node.visibility != 'private':
            self.put(node.visibility)
            self.put(u' ')
        if node.api:
            self.put(u'api ')
        if node.base_type:
            self.visit(node.base_type)
            if node.base_type.name is not None:
                self.put(u' ')
        self.visit(node.declarator.base)
        self.put(u'(')
        self.comma_separated_list(node.declarator.args)
        self.endline(u'):')
        self._visit_indented(node.body)

    def visit_CArgDeclNode(self, node):
        if not isinstance(node.base_type, CSimpleBaseTypeNode) or node.base_type.name is not None:
            self.visit(node.base_type)
            if node.declarator.declared_name():
                self.put(u' ')
        self.visit(node.declarator)
        if node.default is not None:
            self.put(u' = ')
            self.visit(node.default)

    def visit_CImportStatNode(self, node):
        self.startline(u'cimport ')
        self.put(node.module_name)
        if node.as_name:
            self.put(u' as ')
            self.put(node.as_name)
        self.endline()

    def visit_FromCImportStatNode(self, node):
        self.startline(u'from ')
        self.put(node.module_name)
        self.put(u' cimport ')
        first = True
        for pos, name, as_name, kind in node.imported_names:
            assert kind is None
            if first:
                first = False
            else:
                self.put(u', ')
            self.put(name)
            if as_name:
                self.put(u' as ')
                self.put(as_name)
        self.endline()

    def visit_NameNode(self, node):
        self.put(node.name)

    def visit_DecoratorNode(self, node):
        self.startline('@')
        self.visit(node.decorator)
        self.endline()

    def visit_PassStatNode(self, node):
        self.startline(u'pass')
        self.endline()