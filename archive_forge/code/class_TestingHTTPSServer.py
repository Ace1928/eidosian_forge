import ssl
from . import http_server, ssl_certs, test_server
class TestingHTTPSServer(TestingHTTPSServerMixin, http_server.TestingHTTPServer):

    def __init__(self, server_address, request_handler_class, test_case_server, key_file, cert_file):
        TestingHTTPSServerMixin.__init__(self, key_file, cert_file)
        http_server.TestingHTTPServer.__init__(self, server_address, request_handler_class, test_case_server)

    def get_request(self):
        sock, addr = http_server.TestingHTTPServer.get_request(self)
        return self._get_ssl_request(sock, addr)