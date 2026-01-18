import os
import select
import threading
from pyglet import app
from pyglet.app.base import PlatformEventLoop
class XlibEventLoop(PlatformEventLoop):

    def __init__(self):
        super(XlibEventLoop, self).__init__()
        self._notification_device = NotificationDevice()
        self.select_devices = set()
        self.select_devices.add(self._notification_device)

    def notify(self):
        self._notification_device.set()

    def step(self, timeout=None):
        pending_devices = []
        for device in self.select_devices:
            if device.poll():
                pending_devices.append(device)
        if not pending_devices:
            pending_devices, _, _ = select.select(self.select_devices, (), (), timeout)
        if not pending_devices:
            return False
        for device in pending_devices:
            device.select()
        return True