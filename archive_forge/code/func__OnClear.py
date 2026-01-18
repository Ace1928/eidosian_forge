from pidWxDc import PiddleWxDc
from wxPython.wx import *
def _OnClear(self, event):
    """Clears the canvas by emptying the memory buffer, and redrawing"""
    self.MemDc.Clear()
    dc = wxClientDC(self.window)
    dc.Blit(0, 0, self.size[0], self.size[1], self.MemDc, 0, 0, wxCOPY)