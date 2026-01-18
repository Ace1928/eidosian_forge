import re
class NumbaTuplePrinter:

    def __init__(self, val):
        self.val = val

    def to_string(self):
        buf = []
        fields = self.val.type.fields()
        for f in fields:
            buf.append(str(self.val[f.name]))
        return '(%s)' % ', '.join(buf)