from __future__ import annotations
import re
from typing import List, TYPE_CHECKING, Optional, Any
import pyglet
import pyglet.text.layout
from pyglet.gl import GL_TEXTURE0, glActiveTexture, glBindTexture, glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, \
class OrderedListBuilder(ListBuilder):
    format_re = re.compile('(.*?)([1aAiI])(.*)')

    def __init__(self, start, fmt):
        """Create an ordered list with sequentially numbered mark text.

        The format is composed of an optional prefix text, a numbering
        scheme character followed by suffix text. Valid numbering schemes
        are:

        ``1``
            Decimal Arabic
        ``a``
            Lowercase alphanumeric
        ``A``
            Uppercase alphanumeric
        ``i``
            Lowercase Roman
        ``I``
            Uppercase Roman

        Prefix text may typically be ``(`` or ``[`` and suffix text is
        typically ``.``, ``)`` or empty, but either can be any string.

        :Parameters:
            `start` : int
                First list item number.
            `fmt` : str
                Format style, for example ``"1."``.

        """
        self.next_value = start
        self.prefix, self.numbering, self.suffix = self.format_re.match(fmt).groups()
        assert self.numbering in '1aAiI'

    def get_mark(self, value):
        if value is None:
            value = self.next_value
        self.next_value = value + 1
        if self.numbering in 'aA':
            try:
                mark = 'abcdefghijklmnopqrstuvwxyz'[value - 1]
            except ValueError:
                mark = '?'
            if self.numbering == 'A':
                mark = mark.upper()
            return f'{self.prefix}{mark}{self.suffix}'
        elif self.numbering in 'iI':
            try:
                mark = _int_to_roman(value)
            except ValueError:
                mark = '?'
            if self.numbering == 'i':
                mark = mark.lower()
            return f'{self.prefix}{mark}{self.suffix}'
        else:
            return f'{self.prefix}{value}{self.suffix}'