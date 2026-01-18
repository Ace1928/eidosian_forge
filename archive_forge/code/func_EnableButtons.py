import wx
def EnableButtons(self, state=True):
    """Enable the checking-related buttons"""
    if state != self._buttonsEnabled:
        for btn in self.buttons[:-1]:
            btn.Enable(state)
        self._buttonsEnabled = state