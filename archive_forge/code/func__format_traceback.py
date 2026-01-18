import os
from typing import Optional, Tuple, Union
from .util import NO_UTF8, color, supports_ansi
def _format_traceback(self, path, line, fn, text, i, count, highlight):
    template = '{base_indent}{indent} {fn} in {path}:{line}{text}'
    indent = (LINE_EDGE if i == count - 1 else LINE_FORK) + LINE_PATH * i
    if self.tb_base and self.tb_base in path:
        path = path.rsplit(self.tb_base, 1)[1]
    text = self._format_user_error(text, i, highlight) if i == count - 1 else ''
    if self.supports_ansi:
        fn = color(fn, bold=True)
        path = color(path, underline=True)
    return template.format(base_indent=self.indent, line=line, indent=indent, text=text, fn=fn, path=path)