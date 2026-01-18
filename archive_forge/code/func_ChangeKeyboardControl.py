import xcffib
import struct
import io
def ChangeKeyboardControl(self, value_mask, value_list, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', value_mask))
    if value_mask & KB.KeyClickPercent:
        key_click_percent = value_list.pop(0)
        buf.write(struct.pack('=i', key_click_percent))
    if value_mask & KB.BellPercent:
        bell_percent = value_list.pop(0)
        buf.write(struct.pack('=i', bell_percent))
    if value_mask & KB.BellPitch:
        bell_pitch = value_list.pop(0)
        buf.write(struct.pack('=i', bell_pitch))
    if value_mask & KB.BellDuration:
        bell_duration = value_list.pop(0)
        buf.write(struct.pack('=i', bell_duration))
    if value_mask & KB.Led:
        led = value_list.pop(0)
        buf.write(struct.pack('=I', led))
    if value_mask & KB.LedMode:
        led_mode = value_list.pop(0)
        buf.write(struct.pack('=I', led_mode))
    if value_mask & KB.Key:
        key = value_list.pop(0)
        buf.write(struct.pack('=I', key))
    if value_mask & KB.AutoRepeatMode:
        auto_repeat_mode = value_list.pop(0)
        buf.write(struct.pack('=I', auto_repeat_mode))
    return self.send_request(102, buf, is_checked=is_checked)