import xcffib
import struct
import io
from . import xproto
def ChangeAlarm(self, id, value_mask, value_list, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', id, value_mask))
    if value_mask & CA.Counter:
        counter = value_list.pop(0)
        buf.write(struct.pack('=I', counter))
    if value_mask & CA.ValueType:
        valueType = value_list.pop(0)
        buf.write(struct.pack('=I', valueType))
    if value_mask & CA.Value:
        value = value_list.pop(0)
        buf.write(value.pack() if hasattr(value, 'pack') else INT64.synthetic(*value).pack())
    if value_mask & CA.TestType:
        testType = value_list.pop(0)
        buf.write(struct.pack('=I', testType))
    if value_mask & CA.Delta:
        delta = value_list.pop(0)
        buf.write(delta.pack() if hasattr(delta, 'pack') else INT64.synthetic(*delta).pack())
    if value_mask & CA.Events:
        events = value_list.pop(0)
        buf.write(struct.pack('=I', events))
    return self.send_request(9, buf, is_checked=is_checked)