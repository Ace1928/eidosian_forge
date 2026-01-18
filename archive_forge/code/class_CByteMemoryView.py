import re
import traitlets
import datetime as dt
class CByteMemoryView(ByteMemoryView):
    """A casting version of the byte memory view trait."""

    def validate(self, obj, value):
        if isinstance(value, memoryview) and value.format == 'B':
            return value
        try:
            mv = memoryview(value)
            if mv.format != 'B':
                mv = mv.cast('B')
            return mv
        except Exception:
            self.error(obj, value)