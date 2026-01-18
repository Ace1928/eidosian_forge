import unicodedata
def _star(self, rule, vs=None):
    vs = vs or []
    while not self.failed:
        p = self.pos
        rule()
        if self.failed:
            self._rewind(p)
            break
        vs.append(self.val)
    self._succeed(vs)