import wx
def OnIgnoreAll(self, evt):
    """Callback for the "ignore all" button."""
    self._checker.ignore_always()
    self.Advance()