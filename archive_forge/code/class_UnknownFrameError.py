class UnknownFrameError(ValueError):
    """
    An frame of unknown type was received.
    """

    def __init__(self, frame_type, length):
        self.frame_type = frame_type
        self.length = length

    def __str__(self):
        return 'UnknownFrameError: Unknown frame type 0x%X received, length %d bytes' % (self.frame_type, self.length)