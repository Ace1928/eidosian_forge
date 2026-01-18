from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class EventField(ValueField):
    structcode = None

    def pack_value(self, value):
        if not isinstance(value, Event):
            raise BadDataError('%s is not an Event for field %s' % (value, self.name))
        return (value._binary, None, None)

    def parse_binary_value(self, data, display, length, format):
        from Xlib.protocol import event
        estruct = display.event_classes.get(_bytes_item(data[0]) & 127, event.AnyEvent)
        return (estruct(display=display, binarydata=data[:32]), data[32:])