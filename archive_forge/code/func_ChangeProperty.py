import xcffib
import struct
import io
def ChangeProperty(self, mode, window, property, type, format, data_len, data, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIIIB3xI', mode, window, property, type, format, data_len))
    buf.write(xcffib.pack_list(data, 'c'))
    return self.send_request(18, buf, is_checked=is_checked)