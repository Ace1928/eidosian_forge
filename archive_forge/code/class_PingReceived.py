import binascii
from .settings import ChangedSetting, _setting_code_from_int
class PingReceived(Event):
    """
    The PingReceived event is fired whenever a PING is received. It contains
    the 'opaque data' of the PING frame. A ping acknowledgment with the same
    'opaque data' is automatically emitted after receiving a ping.

    .. versionadded:: 3.1.0
    """

    def __init__(self):
        self.ping_data = None

    def __repr__(self):
        return '<PingReceived ping_data:%s>' % (_bytes_representation(self.ping_data),)