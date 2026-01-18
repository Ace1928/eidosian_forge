import re
from mako import exceptions
def _flush_adjusted_lines(self):
    stripspace = None
    self._reset_multi_line_flags()
    for entry in self.line_buffer:
        if self._in_multi_line(entry):
            self.stream.write(entry + '\n')
        else:
            entry = entry.expandtabs()
            if stripspace is None and re.search('^[ \\t]*[^# \\t]', entry):
                stripspace = re.match('^([ \\t]*)', entry).group(1)
            self.stream.write(self._indent_line(entry, stripspace) + '\n')
    self.line_buffer = []
    self._reset_multi_line_flags()