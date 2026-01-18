import gtk
def _onReplace(self, *args):
    print(['Replace'])
    repl = self._getRepl()
    self._checker.replace(repl)
    self._advance()