import ssl
from . import http_server, ssl_certs, test_server
def _get_ssl_request(self, sock, addr):
    """Wrap the socket with SSL"""
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    if self.cert_file:
        ssl_context.load_cert_chain(self.cert_file, self.key_file)
    ssl_sock = ssl_context.wrap_socket(sock=sock, server_side=True, do_handshake_on_connect=False)
    return (ssl_sock, addr)