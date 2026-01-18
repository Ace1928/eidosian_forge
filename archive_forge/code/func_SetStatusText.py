from pidWxDc import PiddleWxDc
from wxPython.wx import *
def SetStatusText(self, s):
    self.text = s
    self.redraw()