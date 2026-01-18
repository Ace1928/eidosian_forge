import xcffib
import struct
import io
from . import xproto
class syncExtension(xcffib.Extension):

    def Initialize(self, desired_major_version, desired_minor_version, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xBB', desired_major_version, desired_minor_version))
        return self.send_request(0, buf, InitializeCookie, is_checked=is_checked)

    def ListSystemCounters(self, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        return self.send_request(1, buf, ListSystemCountersCookie, is_checked=is_checked)

    def CreateCounter(self, id, initial_value, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', id))
        buf.write(initial_value.pack() if hasattr(initial_value, 'pack') else INT64.synthetic(*initial_value).pack())
        return self.send_request(2, buf, is_checked=is_checked)

    def DestroyCounter(self, counter, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', counter))
        return self.send_request(6, buf, is_checked=is_checked)

    def QueryCounter(self, counter, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', counter))
        return self.send_request(5, buf, QueryCounterCookie, is_checked=is_checked)

    def Await(self, wait_list_len, wait_list, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        buf.write(xcffib.pack_list(wait_list, WAITCONDITION))
        return self.send_request(7, buf, is_checked=is_checked)

    def ChangeCounter(self, counter, amount, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', counter))
        buf.write(amount.pack() if hasattr(amount, 'pack') else INT64.synthetic(*amount).pack())
        return self.send_request(4, buf, is_checked=is_checked)

    def SetCounter(self, counter, value, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', counter))
        buf.write(value.pack() if hasattr(value, 'pack') else INT64.synthetic(*value).pack())
        return self.send_request(3, buf, is_checked=is_checked)

    def CreateAlarm(self, id, value_mask, value_list, is_checked=False):
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
        return self.send_request(8, buf, is_checked=is_checked)

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

    def DestroyAlarm(self, alarm, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', alarm))
        return self.send_request(11, buf, is_checked=is_checked)

    def QueryAlarm(self, alarm, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', alarm))
        return self.send_request(10, buf, QueryAlarmCookie, is_checked=is_checked)

    def SetPriority(self, id, priority, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIi', id, priority))
        return self.send_request(12, buf, is_checked=is_checked)

    def GetPriority(self, id, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', id))
        return self.send_request(13, buf, GetPriorityCookie, is_checked=is_checked)

    def CreateFence(self, drawable, fence, initially_triggered, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xIIB', drawable, fence, initially_triggered))
        return self.send_request(14, buf, is_checked=is_checked)

    def TriggerFence(self, fence, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', fence))
        return self.send_request(15, buf, is_checked=is_checked)

    def ResetFence(self, fence, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', fence))
        return self.send_request(16, buf, is_checked=is_checked)

    def DestroyFence(self, fence, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', fence))
        return self.send_request(17, buf, is_checked=is_checked)

    def QueryFence(self, fence, is_checked=True):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2xI', fence))
        return self.send_request(18, buf, QueryFenceCookie, is_checked=is_checked)

    def AwaitFence(self, fence_list_len, fence_list, is_checked=False):
        buf = io.BytesIO()
        buf.write(struct.pack('=xx2x'))
        buf.write(xcffib.pack_list(fence_list, 'I'))
        return self.send_request(19, buf, is_checked=is_checked)