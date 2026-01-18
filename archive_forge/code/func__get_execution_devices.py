import cupy
import numpy
def _get_execution_devices(self, dist_args):
    devices = set()
    for _, arg in dist_args:
        for dev in arg._chunks:
            devices.add(dev)
    return devices