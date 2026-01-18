from antlr4.Token import Token
def _loadString(self):
    self._index = 0
    self.data = [ord(c) for c in self.strdata]
    self._size = len(self.data)