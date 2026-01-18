import functools
import threading
from contextlib import contextmanager
from .driver import driver, USE_NV_BINDING
class _DeviceList(object):

    def __getattr__(self, attr):
        if attr == 'lst':
            numdev = driver.get_device_count()
            gpus = [_DeviceContextManager(driver.get_device(devid)) for devid in range(numdev)]
            self.lst = gpus
            return gpus
        return super(_DeviceList, self).__getattr__(attr)

    def __getitem__(self, devnum):
        """
        Returns the context manager for device *devnum*.
        """
        return self.lst[devnum]

    def __str__(self):
        return ', '.join([str(d) for d in self.lst])

    def __iter__(self):
        return iter(self.lst)

    def __len__(self):
        return len(self.lst)

    @property
    def current(self):
        """Returns the active device or None if there's no active device
        """
        with driver.get_active_context() as ac:
            devnum = ac.devnum
            if devnum is not None:
                return self[devnum]