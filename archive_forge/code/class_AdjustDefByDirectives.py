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
class AdjustDefByDirectives(CythonTransform, SkipDeclarations):
    """
    Adjust function and class definitions by the decorator directives:

    @cython.cfunc
    @cython.cclass
    @cython.ccall
    @cython.inline
    @cython.nogil
    """
    converts_to_cclass = ('cclass', 'total_ordering', 'dataclasses.dataclass')

    def visit_ModuleNode(self, node):
        self.directives = node.directives
        self.in_py_class = False
        self.visitchildren(node)
        return node

    def visit_CompilerDirectivesNode(self, node):
        old_directives = self.directives
        self.directives = node.directives
        self.visitchildren(node)
        self.directives = old_directives
        return node

    def visit_DefNode(self, node):
        modifiers = []
        if 'inline' in self.directives:
            modifiers.append('inline')
        nogil = self.directives.get('nogil')
        with_gil = self.directives.get('with_gil')
        except_val = self.directives.get('exceptval')
        has_explicit_exc_clause = False if except_val is None else True
        return_type_node = self.directives.get('returns')
        if return_type_node is None and self.directives['annotation_typing']:
            return_type_node = node.return_type_annotation
            if return_type_node is not None and except_val is None:
                except_val = (None, True)
        elif except_val is None:
            except_val = (None, True if return_type_node else False)
        if 'ccall' in self.directives:
            if 'cfunc' in self.directives:
                error(node.pos, 'cfunc and ccall directives cannot be combined')
            if with_gil:
                error(node.pos, "ccall functions cannot be declared 'with_gil'")
            node = node.as_cfunction(overridable=True, modifiers=modifiers, nogil=nogil, returns=return_type_node, except_val=except_val, has_explicit_exc_clause=has_explicit_exc_clause)
            return self.visit(node)
        if 'cfunc' in self.directives:
            if self.in_py_class:
                error(node.pos, 'cfunc directive is not allowed here')
            else:
                node = node.as_cfunction(overridable=False, modifiers=modifiers, nogil=nogil, with_gil=with_gil, returns=return_type_node, except_val=except_val, has_explicit_exc_clause=has_explicit_exc_clause)
                return self.visit(node)
        if 'inline' in modifiers:
            error(node.pos, "Python functions cannot be declared 'inline'")
        if nogil:
            error(node.pos, "Python functions cannot be declared 'nogil'")
        if with_gil:
            error(node.pos, "Python functions cannot be declared 'with_gil'")
        self.visitchildren(node)
        return node

    def visit_LambdaNode(self, node):
        return node

    def visit_PyClassDefNode(self, node):
        if any((directive in self.directives for directive in self.converts_to_cclass)):
            node = node.as_cclass()
            return self.visit(node)
        else:
            old_in_pyclass = self.in_py_class
            self.in_py_class = True
            self.visitchildren(node)
            self.in_py_class = old_in_pyclass
            return node

    def visit_CClassDefNode(self, node):
        old_in_pyclass = self.in_py_class
        self.in_py_class = False
        self.visitchildren(node)
        self.in_py_class = old_in_pyclass
        return node