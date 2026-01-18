import numpy as np
from collections import namedtuple
class FakeDeviceList:
    """
    This stub implements a device list containing a single GPU. It also
    keeps track of the GPU status, i.e. whether the context is closed or not,
    which may have been set by the user calling reset()
    """

    def __init__(self):
        self.lst = (FakeCUDAContext(0),)
        self.closed = False

    def __getitem__(self, devnum):
        self.closed = False
        return self.lst[devnum]

    def __str__(self):
        return ', '.join([str(d) for d in self.lst])

    def __iter__(self):
        return iter(self.lst)

    def __len__(self):
        return len(self.lst)

    @property
    def current(self):
        if self.closed:
            return None
        return self.lst[0]