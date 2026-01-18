import xcffib
import struct
import io
def GrabPointer(self, owner_events, grab_window, event_mask, pointer_mode, keyboard_mode, confine_to, cursor, time, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIHBBIII', owner_events, grab_window, event_mask, pointer_mode, keyboard_mode, confine_to, cursor, time))
    return self.send_request(26, buf, GrabPointerCookie, is_checked=is_checked)