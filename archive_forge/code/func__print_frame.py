from os import path
import sys
import traceback
from cupy.cuda import memory_hook
def _print_frame(self, memory_frame, depth=0, file=sys.stdout):
    indent = ' ' * (depth * 2)
    st = memory_frame.stackframe
    used_bytes, acquired_bytes = memory_frame.humanized_bytes()
    line = '%s%s:%s:%s (%s, %s)\n' % (indent, st.filename, st.lineno, st.name, used_bytes, acquired_bytes)
    file.write(line)
    for child in memory_frame.children:
        self._print_frame(child, depth=depth + 1, file=file)