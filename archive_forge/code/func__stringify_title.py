from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _stringify_title(self, title, options):
    lines = []
    lpad, rpad = self._get_padding_widths(options)
    if options['border']:
        if options['vrules'] == ALL:
            options['vrules'] = FRAME
            lines.append(self._stringify_hrule(options, 'top_'))
            options['vrules'] = ALL
        elif options['vrules'] == FRAME:
            lines.append(self._stringify_hrule(options, 'top_'))
    bits = []
    endpoint = options['vertical_char'] if options['vrules'] in (ALL, FRAME) and options['border'] else ' '
    bits.append(endpoint)
    title = ' ' * lpad + title + ' ' * rpad
    bits.append(self._justify(title, len(self._hrule) - 2, 'c'))
    bits.append(endpoint)
    lines.append(''.join(bits))
    return '\n'.join(lines)