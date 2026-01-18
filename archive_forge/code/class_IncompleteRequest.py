from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
class IncompleteRequest(FakeRequest):
    """A request-like object that always expects to read more bytes."""

    def next_read_size(self):
        return 1