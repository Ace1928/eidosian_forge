from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
def _fake_make_request(self, transport, write_func, bytes, rcp):
    request = FakeRequest(transport, write_func)
    request.accept_bytes(bytes)
    self.request = request
    return request