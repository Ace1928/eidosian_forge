import wx
def OnReplaceAll(self, evt):
    """Callback for the "replace all" button."""
    repl = self.GetRepl()
    self._checker.replace_always(repl)
    self.Advance()