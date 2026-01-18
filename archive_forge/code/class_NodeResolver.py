from suds import *
from suds.sax import splitPrefix, Namespace
from suds.sudsobject import Object
from suds.xsd.query import BlindQuery, TypeQuery, qualify
import re
from logging import getLogger
class NodeResolver(TreeResolver):
    """
    The node resolver is a I{stateful} XML document resolver
    used to resolve each node in a tree.  As such, it mirrors
    the tree structure to ensure that nodes are resolved in
    context.
    """

    def __init__(self, schema):
        """
        @param schema: A schema object.
        @type schema: L{xsd.schema.Schema}
        """
        TreeResolver.__init__(self, schema)

    def find(self, node, resolved=False, push=True):
        """
        @param node: An xml node to be resolved.
        @type node: L{sax.element.Element}
        @param resolved: A flag indicating that the fully resolved type should be
            returned.
        @type resolved: boolean
        @param push: Indicates that the resolved type should be
            pushed onto the stack.
        @type push: boolean
        @return: The found schema I{type}
        @rtype: L{xsd.sxbase.SchemaObject}
        """
        name = node.name
        parent = self.top().resolved
        if parent is None:
            result, ancestry = self.query(name, node)
        else:
            result, ancestry = self.getchild(name, parent)
        known = self.known(node)
        if result is None:
            return result
        if push:
            frame = Frame(result, resolved=known, ancestry=ancestry)
            pushed = self.push(frame)
        if resolved:
            result = result.resolve()
        return result

    def findattr(self, name, resolved=True):
        """
        Find an attribute type definition.
        @param name: An attribute name.
        @type name: basestring
        @param resolved: A flag indicating that the fully resolved type should be
            returned.
        @type resolved: boolean
        @return: The found schema I{type}
        @rtype: L{xsd.sxbase.SchemaObject}
        """
        name = '@%s' % name
        parent = self.top().resolved
        if parent is None:
            return None
        else:
            result, ancestry = self.getchild(name, parent)
        if result is None:
            return result
        if resolved:
            result = result.resolve()
        return result

    def query(self, name, node):
        """Blindly query the schema by name."""
        log.debug('searching schema for (%s)', name)
        qref = qualify(name, node, node.namespace())
        query = BlindQuery(qref)
        result = query.execute(self.schema)
        return (result, [])

    def known(self, node):
        """Resolve type referenced by @xsi:type."""
        ref = node.get('type', Namespace.xsins)
        if ref is None:
            return None
        qref = qualify(ref, node, node.namespace())
        query = BlindQuery(qref)
        return query.execute(self.schema)