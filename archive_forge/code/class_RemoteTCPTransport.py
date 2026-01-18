imported from breezy.bzr.smart.
from io import BytesIO
from .. import config, debug, errors, trace, transport, urlutils
from ..bzr import remote
from ..bzr.smart import client, medium
class RemoteTCPTransport(RemoteTransport):
    """Connection to smart server over plain tcp.

    This is essentially just a factory to get 'RemoteTransport(url,
        SmartTCPClientMedium).
    """

    def _build_medium(self):
        client_medium = medium.SmartTCPClientMedium(self._parsed_url.host, self._parsed_url.port, self.base)
        return (client_medium, None)