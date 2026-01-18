import binascii
from .settings import ChangedSetting, _setting_code_from_int
class UnknownFrameReceived(Event):
    """
    The UnknownFrameReceived event is fired when the remote peer sends a frame
    that hyper-h2 does not understand. This occurs primarily when the remote
    peer is employing HTTP/2 extensions that hyper-h2 doesn't know anything
    about.

    RFC 7540 requires that HTTP/2 implementations ignore these frames. hyper-h2
    does so. However, this event is fired to allow implementations to perform
    special processing on those frames if needed (e.g. if the implementation
    is capable of handling the frame itself).

    .. versionadded:: 2.7.0
    """

    def __init__(self):
        self.frame = None

    def __repr__(self):
        return '<UnknownFrameReceived>'