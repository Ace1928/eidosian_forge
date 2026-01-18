import http.client
import http.server
import threading
from oslo_utils import units
class RemoteImageHandler(http.server.BaseHTTPRequestHandler):

    def do_HEAD(self):
        """
        Respond to an image HEAD request fake metadata
        """
        if 'images' in self.path:
            self.send_response(http.client.OK)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', FIVE_KB)
            self.end_headers()
            return
        else:
            self.send_error(http.client.NOT_FOUND, 'File Not Found: %s' % self.path)
            return

    def do_GET(self):
        """
        Respond to an image GET request with fake image content.
        """
        if 'images' in self.path:
            self.send_response(http.client.OK)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', FIVE_KB)
            self.end_headers()
            image_data = b'*' * FIVE_KB
            self.wfile.write(image_data)
            self.wfile.close()
            return
        else:
            self.send_error(http.client.NOT_FOUND, 'File Not Found: %s' % self.path)
            return

    def log_message(self, format, *args):
        """
        Simple override to prevent writing crap to stderr...
        """
        pass