import binascii
class ObjectMissing(Exception):
    """Indicates that a requested object is missing."""

    def __init__(self, sha, *args, **kwargs) -> None:
        Exception.__init__(self, '%s is not in the pack' % sha)