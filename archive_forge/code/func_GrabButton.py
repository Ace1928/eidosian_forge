import xcffib
import struct
import io
def GrabButton(self, owner_events, grab_window, event_mask, pointer_mode, keyboard_mode, confine_to, cursor, button, modifiers, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIHBBIIBxH', owner_events, grab_window, event_mask, pointer_mode, keyboard_mode, confine_to, cursor, button, modifiers))
    return self.send_request(28, buf, is_checked=is_checked)