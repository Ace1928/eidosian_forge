import h2.errors
class FrameTooLargeError(ProtocolError):
    """
    The frame that we tried to send or that we received was too large.
    """
    error_code = h2.errors.ErrorCodes.FRAME_SIZE_ERROR