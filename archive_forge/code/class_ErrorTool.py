import cherrypy
from cherrypy._helper import expose
from cherrypy.lib import cptools, encoding, static, jsontools
from cherrypy.lib import sessions as _sessions, xmlrpcutil as _xmlrpc
from cherrypy.lib import caching as _caching
from cherrypy.lib import auth_basic, auth_digest
class ErrorTool(Tool):
    """Tool which is used to replace the default request.error_response."""

    def __init__(self, callable, name=None):
        Tool.__init__(self, None, callable, name)

    def _wrapper(self):
        self.callable(**self._merged_args())

    def _setup(self):
        """Hook this tool into cherrypy.request.

        The standard CherryPy request object will automatically call this
        method when the tool is "turned on" in config.
        """
        cherrypy.serving.request.error_response = self._wrapper