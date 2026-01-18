import h2.errors
class NoSuchStreamError(ProtocolError):
    """
    A stream-specific action referenced a stream that does not exist.

    .. versionchanged:: 2.0.0
       Became a subclass of :class:`ProtocolError
       <h2.exceptions.ProtocolError>`
    """

    def __init__(self, stream_id):
        self.stream_id = stream_id