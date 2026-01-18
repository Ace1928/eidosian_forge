from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
class FakeRequest:

    def __init__(self, transport, write_func):
        self.transport = transport
        self.write_func = write_func
        self.accepted_bytes = b''

    def accept_bytes(self, bytes):
        self.accepted_bytes = bytes
        self.write_func(b'got bytes: ' + bytes)

    def next_read_size(self):
        return 0