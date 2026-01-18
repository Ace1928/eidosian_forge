from __future__ import annotations
import warnings
from typing import Sequence
from zope.interface import Attribute, Interface, implementer
from incremental import Version
from twisted.python.compat import nativeString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.reflect import prefixedMethodNames
from twisted.web._responses import FORBIDDEN, NOT_FOUND
from twisted.web.error import UnsupportedMethod
class IResource(Interface):
    """
    A web resource.
    """
    isLeaf = Attribute('\n        Signal if this IResource implementor is a "leaf node" or not. If True,\n        getChildWithDefault will not be called on this Resource.\n        ')

    def getChildWithDefault(name, request):
        """
        Return a child with the given name for the given request.
        This is the external interface used by the Resource publishing
        machinery. If implementing IResource without subclassing
        Resource, it must be provided. However, if subclassing Resource,
        getChild overridden instead.

        @param name: A single path component from a requested URL.  For example,
            a request for I{http://example.com/foo/bar} will result in calls to
            this method with C{b"foo"} and C{b"bar"} as values for this
            argument.
        @type name: C{bytes}

        @param request: A representation of all of the information about the
            request that is being made for this child.
        @type request: L{twisted.web.server.Request}
        """

    def putChild(path: bytes, child: 'IResource') -> None:
        """
        Put a child L{IResource} implementor at the given path.

        @param path: A single path component, to be interpreted relative to the
            path this resource is found at, at which to put the given child.
            For example, if resource A can be found at I{http://example.com/foo}
            then a call like C{A.putChild(b"bar", B)} will make resource B
            available at I{http://example.com/foo/bar}.

            The path component is I{not} URL-encoded -- pass C{b'foo bar'}
            rather than C{b'foo%20bar'}.
        """

    def render(request):
        """
        Render a request. This is called on the leaf resource for a request.

        @return: Either C{server.NOT_DONE_YET} to indicate an asynchronous or a
            C{bytes} instance to write as the response to the request.  If
            C{NOT_DONE_YET} is returned, at some point later (for example, in a
            Deferred callback) call C{request.write(b"<html>")} to write data to
            the request, and C{request.finish()} to send the data to the
            browser.

        @raise twisted.web.error.UnsupportedMethod: If the HTTP verb
            requested is not supported by this resource.
        """