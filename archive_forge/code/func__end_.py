import unicodedata
def _end_(self):
    if self.pos == self.end:
        self._succeed(None)
    else:
        self._fail()