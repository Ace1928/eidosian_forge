from pidWxDc import PiddleWxDc
from wxPython.wx import *
def _OnLeaveWindow(self, event):
    if self.interactive == false:
        return None
    self.onLeaveWindow(self)