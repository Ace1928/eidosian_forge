import socket
from .base import StatsClientBase, PipelineBase
class UnixSocketStatsClient(StreamClientBase):
    """Unix domain socket version of StatsClient."""

    def __init__(self, socket_path, prefix=None, timeout=None):
        """Create a new client."""
        self._socket_path = socket_path
        self._timeout = timeout
        self._prefix = prefix
        self._sock = None

    def connect(self):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(self._timeout)
        self._sock.connect(self._socket_path)