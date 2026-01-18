import unicodedata
def _anything_(self):
    if self.pos < self.end:
        self._succeed(self.msg[self.pos], self.pos + 1)
    else:
        self._fail()