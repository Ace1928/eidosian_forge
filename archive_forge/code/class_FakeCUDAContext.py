import numpy as np
from collections import namedtuple
class FakeCUDAContext:
    """
    This stub implements functionality only for simulating a single GPU
    at the moment.
    """

    def __init__(self, device_id):
        self._device_id = device_id
        self._device = FakeCUDADevice()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __str__(self):
        return '<Managed Device {self.id}>'.format(self=self)

    @property
    def id(self):
        return self._device_id

    @property
    def device(self):
        return self._device

    @property
    def compute_capability(self):
        return (5, 2)

    def reset(self):
        pass

    def get_memory_info(self):
        """
        Cross-platform free / total host memory is hard without external
        dependencies, e.g. `psutil` - so return infinite memory to maintain API
        type compatibility
        """
        return _MemoryInfo(float('inf'), float('inf'))

    def memalloc(self, sz):
        """
        Allocates memory on the simulated device
        At present, there is no division between simulated
        host memory and simulated device memory.
        """
        return np.ndarray(sz, dtype='u1')

    def memhostalloc(self, sz, mapped=False, portable=False, wc=False):
        """Allocates memory on the host"""
        return self.memalloc(sz)