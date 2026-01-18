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
class DecoratorTransform(ScopeTrackingTransform, SkipDeclarations):
    """
    Transforms method decorators in cdef classes into nested calls or properties.

    Python-style decorator properties are transformed into a PropertyNode
    with up to the three getter, setter and deleter DefNodes.
    The functional style isn't supported yet.
    """
    _properties = None
    _map_property_attribute = {'getter': EncodedString('__get__'), 'setter': EncodedString('__set__'), 'deleter': EncodedString('__del__')}.get

    def visit_CClassDefNode(self, node):
        if self._properties is None:
            self._properties = []
        self._properties.append({})
        node = super(DecoratorTransform, self).visit_CClassDefNode(node)
        self._properties.pop()
        return node

    def visit_PropertyNode(self, node):
        level = 2 if isinstance(node.pos[0], str) else 0
        warning(node.pos, "'property %s:' syntax is deprecated, use '@property'" % node.name, level)
        return node

    def visit_CFuncDefNode(self, node):
        node = self.visit_FuncDefNode(node)
        if not node.decorators:
            return node
        elif self.scope_type != 'cclass' or self.scope_node.visibility != 'extern':
            if not (len(node.decorators) == 1 and node.decorators[0].decorator.is_name and (node.decorators[0].decorator.name == 'staticmethod')):
                error(node.decorators[0].pos, 'Cdef functions cannot take arbitrary decorators.')
            return node
        ret_node = node
        decorator_node = self._find_property_decorator(node)
        if decorator_node:
            if decorator_node.decorator.is_name:
                name = node.declared_name()
                if name:
                    ret_node = self._add_property(node, name, decorator_node)
            else:
                error(decorator_node.pos, 'C property decorator can only be @property')
        if node.decorators:
            return self._reject_decorated_property(node, node.decorators[0])
        return ret_node

    def visit_DefNode(self, node):
        scope_type = self.scope_type
        node = self.visit_FuncDefNode(node)
        if scope_type != 'cclass' or not node.decorators:
            return node
        decorator_node = self._find_property_decorator(node)
        if decorator_node is not None:
            decorator = decorator_node.decorator
            if decorator.is_name:
                return self._add_property(node, node.name, decorator_node)
            else:
                handler_name = self._map_property_attribute(decorator.attribute)
                if handler_name:
                    if decorator.obj.name != node.name:
                        error(decorator_node.pos, "Mismatching property names, expected '%s', got '%s'" % (decorator.obj.name, node.name))
                    elif len(node.decorators) > 1:
                        return self._reject_decorated_property(node, decorator_node)
                    else:
                        return self._add_to_property(node, handler_name, decorator_node)
        for decorator in node.decorators:
            func = decorator.decorator
            if func.is_name:
                node.is_classmethod |= func.name == 'classmethod'
                node.is_staticmethod |= func.name == 'staticmethod'
        decs = node.decorators
        node.decorators = None
        return self.chain_decorators(node, decs, node.name)

    def _find_property_decorator(self, node):
        properties = self._properties[-1]
        for decorator_node in node.decorators[::-1]:
            decorator = decorator_node.decorator
            if decorator.is_name and decorator.name == 'property':
                return decorator_node
            elif decorator.is_attribute and decorator.obj.name in properties:
                return decorator_node
        return None

    @staticmethod
    def _reject_decorated_property(node, decorator_node):
        for deco in node.decorators:
            if deco != decorator_node:
                error(deco.pos, 'Property methods with additional decorators are not supported')
        return node

    def _add_property(self, node, name, decorator_node):
        if len(node.decorators) > 1:
            return self._reject_decorated_property(node, decorator_node)
        node.decorators.remove(decorator_node)
        properties = self._properties[-1]
        is_cproperty = isinstance(node, Nodes.CFuncDefNode)
        body = Nodes.StatListNode(node.pos, stats=[node])
        if is_cproperty:
            if name in properties:
                error(node.pos, 'C property redeclared')
            if 'inline' not in node.modifiers:
                error(node.pos, "C property method must be declared 'inline'")
            prop = Nodes.CPropertyNode(node.pos, doc=node.doc, name=name, body=body)
        elif name in properties:
            prop = properties[name]
            if prop.is_cproperty:
                error(node.pos, 'C property redeclared')
            else:
                node.name = EncodedString('__get__')
                prop.pos = node.pos
                prop.doc = node.doc
                prop.body.stats = [node]
            return None
        else:
            node.name = EncodedString('__get__')
            prop = Nodes.PropertyNode(node.pos, name=name, doc=node.doc, body=body)
        properties[name] = prop
        return prop

    def _add_to_property(self, node, name, decorator):
        properties = self._properties[-1]
        prop = properties[node.name]
        if prop.is_cproperty:
            error(node.pos, 'C property redeclared')
            return None
        node.name = name
        node.decorators.remove(decorator)
        stats = prop.body.stats
        for i, stat in enumerate(stats):
            if stat.name == name:
                stats[i] = node
                break
        else:
            stats.append(node)
        return None

    @staticmethod
    def chain_decorators(node, decorators, name):
        """
        Decorators are applied directly in DefNode and PyClassDefNode to avoid
        reassignments to the function/class name - except for cdef class methods.
        For those, the reassignment is required as methods are originally
        defined in the PyMethodDef struct.

        The IndirectionNode allows DefNode to override the decorator.
        """
        decorator_result = ExprNodes.NameNode(node.pos, name=name)
        for decorator in decorators[::-1]:
            decorator_result = ExprNodes.SimpleCallNode(decorator.pos, function=decorator.decorator, args=[decorator_result])
        name_node = ExprNodes.NameNode(node.pos, name=name)
        reassignment = Nodes.SingleAssignmentNode(node.pos, lhs=name_node, rhs=decorator_result)
        reassignment = Nodes.IndirectionNode([reassignment])
        node.decorator_indirection = reassignment
        return [node, reassignment]