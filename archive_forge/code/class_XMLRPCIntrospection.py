import base64
import xmlrpc.client as xmlrpclib
from urllib.parse import urlparse
from xmlrpc.client import Binary, Boolean, DateTime, Fault
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure, reflect
from twisted.python.compat import nativeString
from twisted.web import http, resource, server
class XMLRPCIntrospection(XMLRPC):
    """
    Implement the XML-RPC Introspection API.

    By default, the methodHelp method returns the 'help' method attribute,
    if it exists, otherwise the __doc__ method attribute, if it exists,
    otherwise the empty string.

    To enable the methodSignature method, add a 'signature' method attribute
    containing a list of lists. See methodSignature's documentation for the
    format. Note the type strings should be XML-RPC types, not Python types.
    """

    def __init__(self, parent):
        """
        Implement Introspection support for an XMLRPC server.

        @param parent: the XMLRPC server to add Introspection support to.
        @type parent: L{XMLRPC}
        """
        XMLRPC.__init__(self)
        self._xmlrpc_parent = parent

    def xmlrpc_listMethods(self):
        """
        Return a list of the method names implemented by this server.
        """
        functions = []
        todo = [(self._xmlrpc_parent, '')]
        while todo:
            obj, prefix = todo.pop(0)
            functions.extend([prefix + name for name in obj.listProcedures()])
            todo.extend([(obj.getSubHandler(name), prefix + name + obj.separator) for name in obj.getSubHandlerPrefixes()])
        return functions
    xmlrpc_listMethods.signature = [['array']]

    def xmlrpc_methodHelp(self, method):
        """
        Return a documentation string describing the use of the given method.
        """
        method = self._xmlrpc_parent.lookupProcedure(method)
        return getattr(method, 'help', None) or getattr(method, '__doc__', None) or ''
    xmlrpc_methodHelp.signature = [['string', 'string']]

    def xmlrpc_methodSignature(self, method):
        """
        Return a list of type signatures.

        Each type signature is a list of the form [rtype, type1, type2, ...]
        where rtype is the return type and typeN is the type of the Nth
        argument. If no signature information is available, the empty
        string is returned.
        """
        method = self._xmlrpc_parent.lookupProcedure(method)
        return getattr(method, 'signature', None) or ''
    xmlrpc_methodSignature.signature = [['array', 'string'], ['string', 'string']]