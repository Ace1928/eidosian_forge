import h2.errors
class UnsupportedFrameError(ProtocolError, KeyError):
    """
    The remote peer sent a frame that is unsupported in this context.

    .. versionadded:: 2.1.0
    """
    pass