import unicodedata
def _choose(self, rules):
    p = self.pos
    for rule in rules[:-1]:
        rule()
        if not self.failed:
            return
        self._rewind(p)
    rules[-1]()