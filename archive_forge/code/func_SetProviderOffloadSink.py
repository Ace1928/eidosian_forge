import xcffib
import struct
import io
from . import xproto
from . import render
def SetProviderOffloadSink(self, provider, sink_provider, config_timestamp, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', provider, sink_provider, config_timestamp))
    return self.send_request(34, buf, is_checked=is_checked)