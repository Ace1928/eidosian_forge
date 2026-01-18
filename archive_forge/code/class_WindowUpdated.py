import binascii
from .settings import ChangedSetting, _setting_code_from_int
class WindowUpdated(Event):
    """
    The WindowUpdated event is fired whenever a flow control window changes
    size. HTTP/2 defines flow control windows for connections and streams: this
    event fires for both connections and streams. The event carries the ID of
    the stream to which it applies (set to zero if the window update applies to
    the connection), and the delta in the window size.
    """

    def __init__(self):
        self.stream_id = None
        self.delta = None

    def __repr__(self):
        return '<WindowUpdated stream_id:%s, delta:%s>' % (self.stream_id, self.delta)