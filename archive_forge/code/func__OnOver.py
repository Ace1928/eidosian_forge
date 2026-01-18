from pidWxDc import PiddleWxDc
from wxPython.wx import *
def _OnOver(self, event):
    if self.interactive == false:
        return None
    if event.GetY() <= self.size[1]:
        self.onOver(self, event.GetX(), event.GetY())