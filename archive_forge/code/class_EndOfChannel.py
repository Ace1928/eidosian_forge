from trio._util import NoPublicConstructor, final
class EndOfChannel(Exception):
    """Raised when trying to receive from a :class:`trio.abc.ReceiveChannel`
    that has no more data to receive.

    This is analogous to an "end-of-file" condition, but for channels.

    """