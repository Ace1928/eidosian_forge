from pidWxDc import PiddleWxDc
from wxPython.wx import *
def _OnQuit(self, event):
    """Closes the canvas.  Call to return control your application"""
    self.window.Close()