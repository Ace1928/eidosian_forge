import struct
import binascii
from .exceptions import (
from .flags import Flag, Flags
class RstStreamFrame(Frame):
    """
    The RST_STREAM frame allows for abnormal termination of a stream. When sent
    by the initiator of a stream, it indicates that they wish to cancel the
    stream or that an error condition has occurred. When sent by the receiver
    of a stream, it indicates that either the receiver is rejecting the stream,
    requesting that the stream be cancelled or that an error condition has
    occurred.
    """
    defined_flags = []
    type = 3
    stream_association = _STREAM_ASSOC_HAS_STREAM

    def __init__(self, stream_id, error_code=0, **kwargs):
        super(RstStreamFrame, self).__init__(stream_id, **kwargs)
        self.error_code = error_code

    def serialize_body(self):
        return _STRUCT_L.pack(self.error_code)

    def parse_body(self, data):
        if len(data) != 4:
            raise InvalidFrameError('RST_STREAM must have 4 byte body: actual length %s.' % len(data))
        try:
            self.error_code = _STRUCT_L.unpack(data)[0]
        except struct.error:
            raise InvalidFrameError('Invalid RST_STREAM body')
        self.body_len = 4