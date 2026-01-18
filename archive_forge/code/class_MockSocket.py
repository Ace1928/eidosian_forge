from cheroot import makefile
class MockSocket:
    """A mock socket."""

    def __init__(self):
        """Initialize :py:class:`MockSocket`."""
        self.messages = []

    def recv_into(self, buf):
        """Simulate ``recv_into`` for Python 3."""
        if not self.messages:
            return 0
        msg = self.messages.pop(0)
        for index, byte in enumerate(msg):
            buf[index] = byte
        return len(msg)

    def recv(self, size):
        """Simulate ``recv`` for Python 2."""
        try:
            return self.messages.pop(0)
        except IndexError:
            return ''

    def send(self, val):
        """Simulate a send."""
        return len(val)