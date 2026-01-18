import xcffib
import struct
import io
def UngrabButton(self, button, grab_window, modifiers, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIH2x', button, grab_window, modifiers))
    return self.send_request(29, buf, is_checked=is_checked)