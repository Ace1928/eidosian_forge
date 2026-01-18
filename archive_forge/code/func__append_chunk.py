import re
import binascii
import email.quoprimime
import email.base64mime
from email.errors import HeaderParseError
from email import charset as _charset
def _append_chunk(self, fws, string):
    self._current_line.push(fws, string)
    if len(self._current_line) > self._maxlen:
        for ch in self._splitchars:
            for i in range(self._current_line.part_count() - 1, 0, -1):
                if ch.isspace():
                    fws = self._current_line[i][0]
                    if fws and fws[0] == ch:
                        break
                prevpart = self._current_line[i - 1][1]
                if prevpart and prevpart[-1] == ch:
                    break
            else:
                continue
            break
        else:
            fws, part = self._current_line.pop()
            if self._current_line._initial_size > 0:
                self.newline()
                if not fws:
                    fws = ' '
            self._current_line.push(fws, part)
            return
        remainder = self._current_line.pop_from(i)
        self._lines.append(str(self._current_line))
        self._current_line.reset(remainder)