import sys
import signal
import time
from timeit import default_timer as clock
import wx
class EventLoopRunner(object):

    def Run(self, time, input_is_ready):
        self.input_is_ready = input_is_ready
        self.evtloop = wx.EventLoop()
        self.timer = EventLoopTimer(self.check_stdin)
        self.timer.Start(time)
        self.evtloop.Run()

    def check_stdin(self):
        if self.input_is_ready():
            self.timer.Stop()
            self.evtloop.Exit()