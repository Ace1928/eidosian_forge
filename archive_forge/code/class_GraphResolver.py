from suds import *
from suds.sax import splitPrefix, Namespace
from suds.sudsobject import Object
from suds.xsd.query import BlindQuery, TypeQuery, qualify
import re
from logging import getLogger
class GraphResolver(TreeResolver):
    """
    The graph resolver is a I{stateful} L{Object} graph resolver
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

    def find(self, name, object, resolved=False, push=True):
        """
        @param name: The name of the object to be resolved.
        @type name: basestring
        @param object: The name's value.
        @type object: (any|L{Object})
        @param resolved: A flag indicating that the fully resolved type
            should be returned.
        @type resolved: boolean
        @param push: Indicates that the resolved type should be
            pushed onto the stack.
        @type push: boolean
        @return: The found schema I{type}
        @rtype: L{xsd.sxbase.SchemaObject}
        """
        known = None
        parent = self.top().resolved
        if parent is None:
            result, ancestry = self.query(name)
        else:
            result, ancestry = self.getchild(name, parent)
        if result is None:
            return None
        if isinstance(object, Object):
            known = self.known(object)
        if push:
            frame = Frame(result, resolved=known, ancestry=ancestry)
            pushed = self.push(frame)
        if resolved:
            if known is None:
                result = result.resolve()
            else:
                result = known
        return result

    def query(self, name):
        """Blindly query the schema by name."""
        log.debug('searching schema for (%s)', name)
        schema = self.schema
        wsdl = self.wsdl()
        if wsdl is None:
            qref = qualify(name, schema.root, schema.tns)
        else:
            qref = qualify(name, wsdl.root, wsdl.tns)
        query = BlindQuery(qref)
        result = query.execute(schema)
        return (result, [])

    def wsdl(self):
        """Get the wsdl."""
        container = self.schema.container
        if container is None:
            return None
        else:
            return container.wsdl

    def known(self, object):
        """Get the type specified in the object's metadata."""
        try:
            md = object.__metadata__
            known = md.sxtype
            return known
        except Exception:
            pass