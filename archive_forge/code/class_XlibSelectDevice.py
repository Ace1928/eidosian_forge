import os
import select
import threading
from pyglet import app
from pyglet.app.base import PlatformEventLoop
class XlibSelectDevice:

    def fileno(self):
        """Get the file handle for ``select()`` for this device.

        :rtype: int
        """
        raise NotImplementedError('abstract')

    def select(self):
        """Perform event processing on the device.

        Called when ``select()`` returns this device in its list of active
        files.
        """
        raise NotImplementedError('abstract')

    def poll(self):
        """Check if the device has events ready to process.

        :rtype: bool
        :return: True if there are events to process, False otherwise.
        """
        return False