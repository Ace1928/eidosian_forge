import h2.errors
class StreamIDTooLowError(ProtocolError):
    """
    An attempt was made to open a stream that had an ID that is lower than the
    highest ID we have seen on this connection.
    """

    def __init__(self, stream_id, max_stream_id):
        self.stream_id = stream_id
        self.max_stream_id = max_stream_id

    def __str__(self):
        return 'StreamIDTooLowError: %d is lower than %d' % (self.stream_id, self.max_stream_id)