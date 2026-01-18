import re
class NumbaUnicodeTypePrinter:

    def __init__(self, val):
        self.val = val

    def to_string(self):
        NULL = 0
        data = self.val['data']
        nitems = self.val['length']
        kind = self.val['kind']
        if data != NULL:
            this_proc = gdb.selected_inferior()
            mem = this_proc.read_memory(int(data), nitems * kind)
            if isinstance(mem, memoryview):
                buf = bytes(mem).decode()
            else:
                buf = mem.decode('utf-8')
        else:
            buf = str(data)
        return "'%s'" % buf