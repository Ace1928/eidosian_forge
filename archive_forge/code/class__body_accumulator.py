from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future.builtins import bytes, chr, dict, int, range, super
import re
import io
from string import ascii_letters, digits, hexdigits
class _body_accumulator(io.StringIO):

    def __init__(self, maxlinelen, eol, *args, **kw):
        super().__init__(*args, **kw)
        self.eol = eol
        self.maxlinelen = self.room = maxlinelen

    def write_str(self, s):
        """Add string s to the accumulated body."""
        self.write(s)
        self.room -= len(s)

    def newline(self):
        """Write eol, then start new line."""
        self.write_str(self.eol)
        self.room = self.maxlinelen

    def write_soft_break(self):
        """Write a soft break, then start a new line."""
        self.write_str('=')
        self.newline()

    def write_wrapped(self, s, extra_room=0):
        """Add a soft line break if needed, then write s."""
        if self.room < len(s) + extra_room:
            self.write_soft_break()
        self.write_str(s)

    def write_char(self, c, is_last_char):
        if not is_last_char:
            self.write_wrapped(c, extra_room=1)
        elif c not in ' \t':
            self.write_wrapped(c)
        elif self.room >= 3:
            self.write(quote(c))
        elif self.room == 2:
            self.write(c)
            self.write_soft_break()
        else:
            self.write_soft_break()
            self.write(quote(c))