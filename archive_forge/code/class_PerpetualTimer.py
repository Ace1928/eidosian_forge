import os
import re
import signal as _signal
import sys
import time
import threading
import _thread
from cherrypy._cpcompat import text_or_bytes
from cherrypy._cpcompat import ntob
class PerpetualTimer(threading.Timer):
    """A responsive subclass of threading.Timer whose run() method repeats.

    Use this timer only when you really need a very interruptible timer;
    this checks its 'finished' condition up to 20 times a second, which can
    results in pretty high CPU usage
    """

    def __init__(self, *args, **kwargs):
        """Override parent constructor to allow 'bus' to be provided."""
        self.bus = kwargs.pop('bus', None)
        super(PerpetualTimer, self).__init__(*args, **kwargs)

    def run(self):
        while True:
            self.finished.wait(self.interval)
            if self.finished.isSet():
                return
            try:
                self.function(*self.args, **self.kwargs)
            except Exception:
                if self.bus:
                    self.bus.log('Error in perpetual timer thread function %r.' % self.function, level=40, traceback=True)
                raise