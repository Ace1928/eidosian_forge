import unicodedata
def _ch(self, ch):
    p = self.pos
    if p < self.end and self.msg[p] == ch:
        self._succeed(ch, self.pos + 1)
    else:
        self._fail()