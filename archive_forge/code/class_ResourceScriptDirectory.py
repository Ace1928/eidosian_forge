import os
import traceback
from io import StringIO
from twisted import copyright
from twisted.python.compat import execfile, networkString
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.web import http, resource, server, static, util
import mygreatresource
class ResourceScriptDirectory(resource.Resource):
    """
    L{ResourceScriptDirectory} is a resource which serves scripts from a
    filesystem directory.  File children of a L{ResourceScriptDirectory} will
    be served using L{ResourceScript}.  Directory children will be served using
    another L{ResourceScriptDirectory}.

    @ivar path: A C{str} giving the filesystem path in which children will be
        looked up.

    @ivar registry: A L{static.Registry} instance which will be used to decide
        how to interpret scripts found as children of this resource.
    """

    def __init__(self, pathname, registry=None):
        resource.Resource.__init__(self)
        self.path = pathname
        self.registry = registry or static.Registry()

    def getChild(self, path, request):
        fn = os.path.join(self.path, path)
        if os.path.isdir(fn):
            return ResourceScriptDirectory(fn, self.registry)
        if os.path.exists(fn):
            return ResourceScript(fn, self.registry)
        return resource._UnsafeNoResource()

    def render(self, request):
        return resource._UnsafeNoResource().render(request)