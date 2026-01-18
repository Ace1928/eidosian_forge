import binascii
from .settings import ChangedSetting, _setting_code_from_int
class RequestReceived(Event):
    """
    The RequestReceived event is fired whenever request headers are received.
    This event carries the HTTP headers for the given request and the stream ID
    of the new stream.

    .. versionchanged:: 2.3.0
       Changed the type of ``headers`` to :class:`HeaderTuple
       <hpack:hpack.HeaderTuple>`. This has no effect on current users.

    .. versionchanged:: 2.4.0
       Added ``stream_ended`` and ``priority_updated`` properties.
    """

    def __init__(self):
        self.stream_id = None
        self.headers = None
        self.stream_ended = None
        self.priority_updated = None

    def __repr__(self):
        return '<RequestReceived stream_id:%s, headers:%s>' % (self.stream_id, self.headers)