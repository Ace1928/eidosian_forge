import struct
import binascii
from .exceptions import (
from .flags import Flag, Flags
class SettingsFrame(Frame):
    """
    The SETTINGS frame conveys configuration parameters that affect how
    endpoints communicate. The parameters are either constraints on peer
    behavior or preferences.

    Settings are not negotiated. Settings describe characteristics of the
    sending peer, which are used by the receiving peer. Different values for
    the same setting can be advertised by each peer. For example, a client
    might set a high initial flow control window, whereas a server might set a
    lower value to conserve resources.
    """
    defined_flags = [Flag('ACK', 1)]
    type = 4
    stream_association = _STREAM_ASSOC_NO_STREAM
    HEADER_TABLE_SIZE = 1
    ENABLE_PUSH = 2
    MAX_CONCURRENT_STREAMS = 3
    INITIAL_WINDOW_SIZE = 4
    MAX_FRAME_SIZE = 5
    MAX_HEADER_LIST_SIZE = 6
    ENABLE_CONNECT_PROTOCOL = 8

    def __init__(self, stream_id=0, settings=None, **kwargs):
        super(SettingsFrame, self).__init__(stream_id, **kwargs)
        if settings and 'ACK' in kwargs.get('flags', ()):
            raise ValueError('Settings must be empty if ACK flag is set.')
        self.settings = settings or {}

    def serialize_body(self):
        return b''.join([_STRUCT_HL.pack(setting & 255, value) for setting, value in self.settings.items()])

    def parse_body(self, data):
        body_len = 0
        for i in range(0, len(data), 6):
            try:
                name, value = _STRUCT_HL.unpack(data[i:i + 6])
            except struct.error:
                raise InvalidFrameError('Invalid SETTINGS body')
            self.settings[name] = value
            body_len += 6
        self.body_len = body_len