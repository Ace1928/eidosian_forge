import logging
from time import gmtime
class ReceivedRoute(object):
    """Holds the information that has been received to one sinks
    about a particular BGP destination.
    """

    def __init__(self, path, peer, filtered=None, timestamp=None):
        assert path and hasattr(peer, 'version_num')
        self.path = path
        self._received_peer = peer
        self.filtered = filtered
        if timestamp:
            self.timestamp = timestamp
        else:
            self.timestamp = gmtime()

    @property
    def received_peer(self):
        return self._received_peer