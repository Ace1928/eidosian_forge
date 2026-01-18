from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import urllib.parse
from wsgiref.handlers import SimpleHandler
from platform import python_implementation
class ServerHandler(SimpleHandler):
    server_software = software_version

    def close(self):
        try:
            self.request_handler.log_request(self.status.split(' ', 1)[0], self.bytes_sent)
        finally:
            SimpleHandler.close(self)