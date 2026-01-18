import socket
import warnings
from io import BytesIO
from amqp.serialization import _write_table
class PLAIN(SASL):
    """PLAIN SASL authentication mechanism.

    See https://tools.ietf.org/html/rfc4616 for details
    """
    mechanism = b'PLAIN'

    def __init__(self, username, password):
        self.username, self.password = (username, password)
    __slots__ = ('username', 'password')

    def start(self, connection):
        if self.username is None or self.password is None:
            return NotImplemented
        login_response = BytesIO()
        login_response.write(b'\x00')
        login_response.write(self.username.encode('utf-8'))
        login_response.write(b'\x00')
        login_response.write(self.password.encode('utf-8'))
        return login_response.getvalue()