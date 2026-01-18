import sys
import queue
import threading
from pyglet import app
from pyglet import clock
from pyglet import event
def _blocking_timer(self):
    timeout = self.idle()
    app.platform_event_loop.set_timer(self._blocking_timer, timeout)