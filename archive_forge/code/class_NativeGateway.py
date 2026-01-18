import logging
import sys
import io
import cheroot.server
import cherrypy
from cherrypy._cperror import format_exc, bare_error
from cherrypy.lib import httputil
from ._cpcompat import tonative
class NativeGateway(cheroot.server.Gateway):
    """Native gateway implementation allowing to bypass WSGI."""
    recursive = False

    def respond(self):
        """Obtain response from CherryPy machinery and then send it."""
        req = self.req
        try:
            local = req.server.bind_addr
            local = (tonative(local[0]), local[1])
            local = httputil.Host(local[0], local[1], '')
            remote = (tonative(req.conn.remote_addr), req.conn.remote_port)
            remote = httputil.Host(remote[0], remote[1], '')
            scheme = tonative(req.scheme)
            sn = cherrypy.tree.script_name(tonative(req.uri or '/'))
            if sn is None:
                self.send_response('404 Not Found', [], [''])
            else:
                app = cherrypy.tree.apps[sn]
                method = tonative(req.method)
                path = tonative(req.path)
                qs = tonative(req.qs or '')
                headers = ((tonative(h), tonative(v)) for h, v in req.inheaders.items())
                rfile = req.rfile
                prev = None
                try:
                    redirections = []
                    while True:
                        request, response = app.get_serving(local, remote, scheme, 'HTTP/1.1')
                        request.multithread = True
                        request.multiprocess = False
                        request.app = app
                        request.prev = prev
                        try:
                            request.run(method, path, qs, tonative(req.request_protocol), headers, rfile)
                            break
                        except cherrypy.InternalRedirect:
                            ir = sys.exc_info()[1]
                            app.release_serving()
                            prev = request
                            if not self.recursive:
                                if ir.path in redirections:
                                    raise RuntimeError('InternalRedirector visited the same URL twice: %r' % ir.path)
                                else:
                                    if qs:
                                        qs = '?' + qs
                                    redirections.append(sn + path + qs)
                            method = 'GET'
                            path = ir.path
                            qs = ir.query_string
                            rfile = io.BytesIO()
                    self.send_response(response.output_status, response.header_list, response.body)
                finally:
                    app.release_serving()
        except Exception:
            tb = format_exc()
            cherrypy.log(tb, 'NATIVE_ADAPTER', severity=logging.ERROR)
            s, h, b = bare_error()
            self.send_response(s, h, b)

    def send_response(self, status, headers, body):
        """Send response to HTTP request."""
        req = self.req
        req.status = status or b'500 Server Error'
        for header, value in headers:
            req.outheaders.append((header, value))
        if req.ready and (not req.sent_headers):
            req.sent_headers = True
            req.send_headers()
        for seg in body:
            req.write(seg)