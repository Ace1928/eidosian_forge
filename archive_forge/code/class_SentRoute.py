import logging
from time import gmtime
class SentRoute(object):
    """Holds the information that has been sent to one or more sinks
    about a particular BGP destination.
    """

    def __init__(self, path, peer, filtered=None, timestamp=None):
        assert path and hasattr(peer, 'version_num')
        self.path = path
        self._sent_peer = peer
        self.filtered = filtered
        if timestamp:
            self.timestamp = timestamp
        else:
            self.timestamp = gmtime()

    @property
    def sent_peer(self):
        return self._sent_peer