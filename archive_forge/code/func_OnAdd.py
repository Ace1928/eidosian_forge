import wx
def OnAdd(self, evt):
    """Callback for the "add" button."""
    self._checker.add()
    self.Advance()