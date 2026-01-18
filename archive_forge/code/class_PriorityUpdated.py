import binascii
from .settings import ChangedSetting, _setting_code_from_int
class PriorityUpdated(Event):
    """
    The PriorityUpdated event is fired whenever a stream sends updated priority
    information. This can occur when the stream is opened, or at any time
    during the stream lifetime.

    This event is purely advisory, and does not need to be acted on.

    .. versionadded:: 2.0.0
    """

    def __init__(self):
        self.stream_id = None
        self.weight = None
        self.depends_on = None
        self.exclusive = None

    def __repr__(self):
        return '<PriorityUpdated stream_id:%s, weight:%s, depends_on:%s, exclusive:%s>' % (self.stream_id, self.weight, self.depends_on, self.exclusive)