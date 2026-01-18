import sys
import signal
import time
from timeit import default_timer as clock
import wx
class EventLoopTimer(wx.Timer):

    def __init__(self, func):
        self.func = func
        wx.Timer.__init__(self)

    def Notify(self):
        self.func()