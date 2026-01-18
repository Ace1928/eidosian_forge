from pidWxDc import PiddleWxDc
from wxPython.wx import *
def _OnPaint(self, event):
    dc = wxPaintDC(self.window)
    dc.Blit(0, 0, self.size[0], self.size[1], self.MemDc, 0, 0, wxCOPY)
    del dc