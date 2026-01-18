import errno
import os
from functools import partial
from threading import Thread
from pyudev._os import pipe, poll
from pyudev._util import eintr_retry_call, ensure_byte_string
from pyudev.device import Device
def _receive_device(self):
    """Receive a single device from the monitor.

        Return the received :class:`Device`, or ``None`` if no device could be
        received.

        """
    while True:
        try:
            device_p = self._libudev.udev_monitor_receive_device(self)
            return Device(self.context, device_p) if device_p else None
        except EnvironmentError as error:
            if error.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return None
            elif error.errno == errno.EINTR:
                continue
            else:
                raise