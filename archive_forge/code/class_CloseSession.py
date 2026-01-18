from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
class CloseSession(RPC):
    """`close-session` RPC. The connection to NETCONF server is also closed."""

    def request(self):
        """Request graceful termination of the NETCONF session, and also close the transport."""
        ret = self._request(new_ele('close-session'))
        self.session.close()
        return ret