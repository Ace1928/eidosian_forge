import binascii
from .settings import ChangedSetting, _setting_code_from_int
class StreamReset(Event):
    """
    The StreamReset event is fired in two situations. The first is when the
    remote party forcefully resets the stream. The second is when the remote
    party has made a protocol error which only affects a single stream. In this
    case, Hyper-h2 will terminate the stream early and return this event.

    .. versionchanged:: 2.0.0
       This event is now fired when Hyper-h2 automatically resets a stream.
    """

    def __init__(self):
        self.stream_id = None
        self.error_code = None
        self.remote_reset = True

    def __repr__(self):
        return '<StreamReset stream_id:%s, error_code:%s, remote_reset:%s>' % (self.stream_id, self.error_code, self.remote_reset)