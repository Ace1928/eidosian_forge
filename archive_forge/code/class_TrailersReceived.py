import binascii
from .settings import ChangedSetting, _setting_code_from_int
class TrailersReceived(Event):
    """
    The TrailersReceived event is fired whenever trailers are received on a
    stream. Trailers are a set of headers sent after the body of the
    request/response, and are used to provide information that wasn't known
    ahead of time (e.g. content-length). This event carries the HTTP header
    fields that form the trailers and the stream ID of the stream on which they
    were received.

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
        return '<TrailersReceived stream_id:%s, headers:%s>' % (self.stream_id, self.headers)