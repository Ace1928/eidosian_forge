import binascii
class GitProtocolError(Exception):
    """Git protocol exception."""

    def __init__(self, *args, **kwargs) -> None:
        Exception.__init__(self, *args, **kwargs)

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.args == other.args