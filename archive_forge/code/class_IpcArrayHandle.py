import math
import functools
import operator
import copy
from ctypes import c_void_p
import numpy as np
import numba
from numba import _devicearray
from numba.cuda.cudadrv import devices
from numba.cuda.cudadrv import driver as _driver
from numba.core import types, config
from numba.np.unsafe.ndarray import to_fixed_tuple
from numba.misc import dummyarray
from numba.np import numpy_support
from numba.cuda.api_util import prepare_shape_strides_dtype
from numba.core.errors import NumbaPerformanceWarning
from warnings import warn
class IpcArrayHandle(object):
    """
    An IPC array handle that can be serialized and transfer to another process
    in the same machine for share a GPU allocation.

    On the destination process, use the *.open()* method to creates a new
    *DeviceNDArray* object that shares the allocation from the original process.
    To release the resources, call the *.close()* method.  After that, the
    destination can no longer use the shared array object.  (Note: the
    underlying weakref to the resource is now dead.)

    This object implements the context-manager interface that calls the
    *.open()* and *.close()* method automatically::

        with the_ipc_array_handle as ipc_array:
            # use ipc_array here as a normal gpu array object
            some_code(ipc_array)
        # ipc_array is dead at this point
    """

    def __init__(self, ipc_handle, array_desc):
        self._array_desc = array_desc
        self._ipc_handle = ipc_handle

    def open(self):
        """
        Returns a new *DeviceNDArray* that shares the allocation from the
        original process.  Must not be used on the original process.
        """
        dptr = self._ipc_handle.open(devices.get_context())
        return DeviceNDArray(gpu_data=dptr, **self._array_desc)

    def close(self):
        """
        Closes the IPC handle to the array.
        """
        self._ipc_handle.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, type, value, traceback):
        self.close()