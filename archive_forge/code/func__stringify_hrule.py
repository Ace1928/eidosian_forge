from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _stringify_hrule(self, options, where: str=''):
    if not options['border'] and (not options['preserve_internal_border']):
        return ''
    lpad, rpad = self._get_padding_widths(options)
    if options['vrules'] in (ALL, FRAME):
        bits = [options[where + 'left_junction_char']]
    else:
        bits = [options['horizontal_char']]
    if not self._field_names:
        bits.append(options[where + 'right_junction_char'])
        return ''.join(bits)
    for field, width in zip(self._field_names, self._widths):
        if options['fields'] and field not in options['fields']:
            continue
        line = (width + lpad + rpad) * options['horizontal_char']
        if self._horizontal_align_char:
            if self._align[field] in ('l', 'c'):
                line = ' ' + self._horizontal_align_char + line[2:]
            if self._align[field] in ('c', 'r'):
                line = line[:-2] + self._horizontal_align_char + ' '
        bits.append(line)
        if options['vrules'] == ALL:
            bits.append(options[where + 'junction_char'])
        else:
            bits.append(options['horizontal_char'])
    if options['vrules'] in (ALL, FRAME):
        bits.pop()
        bits.append(options[where + 'right_junction_char'])
    if options['preserve_internal_border'] and (not options['border']):
        bits = bits[1:-1]
    return ''.join(bits)