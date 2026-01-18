import gtk
def _onReplaceAll(self, *args):
    print(['Replace all'])
    repl = self._getRepl()
    self._checker.replace_always(repl)
    self._advance()