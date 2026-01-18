import os
import select
import threading
from pyglet import app
from pyglet.app.base import PlatformEventLoop
class NotificationDevice(XlibSelectDevice):

    def __init__(self):
        self._sync_file_read, self._sync_file_write = os.pipe()
        self._event = threading.Event()

    def fileno(self):
        return self._sync_file_read

    def set(self):
        self._event.set()
        os.write(self._sync_file_write, b'1')

    def select(self):
        self._event.clear()
        os.read(self._sync_file_read, 1)
        app.platform_event_loop.dispatch_posted_events()

    def poll(self):
        return self._event.is_set()