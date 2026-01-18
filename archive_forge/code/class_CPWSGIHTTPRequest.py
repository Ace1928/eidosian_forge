import sys
import cheroot.wsgi
import cheroot.server
import cherrypy
class CPWSGIHTTPRequest(cheroot.server.HTTPRequest):
    """Wrapper for cheroot.server.HTTPRequest.

    This is a layer, which preserves URI parsing mode like it which was
    before Cheroot v5.8.0.
    """

    def __init__(self, server, conn):
        """Initialize HTTP request container instance.

        Args:
            server (cheroot.server.HTTPServer):
                web server object receiving this request
            conn (cheroot.server.HTTPConnection):
                HTTP connection object for this request
        """
        super(CPWSGIHTTPRequest, self).__init__(server, conn, proxy_mode=True)