import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
class TimerFrame(wx.Frame):

    def __init__(self, func):
        wx.Frame.__init__(self, None, -1)
        self.timer = wx.Timer(self)
        self.timer.Start(poll_interval)
        self.Bind(wx.EVT_TIMER, self.on_timer)
        self.func = func

    def on_timer(self, event):
        self.func()