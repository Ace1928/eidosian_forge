from ...commands import Command
from ...lazy_import import lazy_import
from breezy.bzr.smart.client import _SmartClient
from breezy.transport import get_transport
from breezy import errors
class cmd_ping(Command):
    """Pings a Bazaar smart server.

    This command sends a 'hello' request to the given location using the brz
    smart protocol, and reports the response.
    """
    takes_args = ['location']

    def run(self, location):
        transport = get_transport(location)
        try:
            medium = transport.get_smart_medium()
        except errors.NoSmartMedium as e:
            raise errors.CommandError(str(e))
        client = _SmartClient(medium)
        response, handler = client.call_expecting_body(b'hello')
        handler.cancel_read_body()
        self.outf.write('Response: {!r}\n'.format(response))
        if getattr(handler, 'headers', None) is not None:
            headers = {k.decode('utf-8'): v.decode('utf-8') for k, v in handler.headers.items()}
            self.outf.write('Headers: {!r}\n'.format(headers))