import argparse
from http import server
import socketserver
import webob
from oslo_middleware import healthcheck
class HttpHandler(server.SimpleHTTPRequestHandler):

    def do_GET(self):

        @webob.dec.wsgify
        def dummy_application(req):
            return 'test'
        app = healthcheck.Healthcheck(dummy_application, {'detailed': True})
        req = webob.Request.blank('/healthcheck', accept='text/html', method='GET')
        res = req.get_response(app)
        self.send_response(res.status_code)
        for header_name, header_value in res.headerlist:
            self.send_header(header_name, header_value)
        self.end_headers()
        self.wfile.write(res.body)
        self.wfile.close()