from __future__ import annotations
from . import Extension
from ..treeprocessors import Treeprocessor
from ..util import parseBoolValue
from typing import TYPE_CHECKING, Callable, Any
def _parseHeader(self) -> None:
    """
        Determines language of a code block from shebang line and whether the
        said line should be removed or left in place. If the shebang line
        contains a path (even a single /) then it is assumed to be a real
        shebang line and left alone. However, if no path is given
        (e.i.: `#!python` or `:::python`) then it is assumed to be a mock shebang
        for language identification of a code fragment and removed from the
        code block prior to processing for code highlighting. When a mock
        shebang (e.i: `#!python`) is found, line numbering is turned on. When
        colons are found in place of a shebang (e.i.: `:::python`), line
        numbering is left in the current state - off by default.

        Also parses optional list of highlight lines, like:

            :::python hl_lines="1 3"
        """
    import re
    lines = self.src.split('\n')
    fl = lines.pop(0)
    c = re.compile('\n            (?:(?:^::+)|(?P<shebang>^[#]!)) # Shebang or 2 or more colons\n            (?P<path>(?:/\\w+)*[/ ])?        # Zero or 1 path\n            (?P<lang>[\\w#.+-]*)             # The language\n            \\s*                             # Arbitrary whitespace\n            # Optional highlight lines, single- or double-quote-delimited\n            (hl_lines=(?P<quot>"|\')(?P<hl_lines>.*?)(?P=quot))?\n            ', re.VERBOSE)
    m = c.search(fl)
    if m:
        try:
            self.lang = m.group('lang').lower()
        except IndexError:
            self.lang = None
        if m.group('path'):
            lines.insert(0, fl)
        if self.options['linenos'] is None and m.group('shebang'):
            self.options['linenos'] = True
        self.options['hl_lines'] = parse_hl_lines(m.group('hl_lines'))
    else:
        lines.insert(0, fl)
    self.src = '\n'.join(lines).strip('\n')