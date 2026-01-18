from pidWxDc import PiddleWxDc
from wxPython.wx import *
def _OnClickUp(self, event):
    if self.interactive == false:
        return None
    self.onClickUp(self, event.GetX(), event.GetY())