import collections
import contextlib
import os
import re
import warnings
import numpy as np
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import distribution_lib
from keras.src.backend.common import global_state
def _initialize_with_device_mesh(self, device_mesh):
    if not isinstance(device_mesh, DeviceMesh):
        raise ValueError(f'Expect `mesh` to be an instance of `DeviceMesh`. Received: mesh={device_mesh} (of type {type(device_mesh)})')
    super().__init__(device_mesh)
    if self.device_mesh.devices.ndim != 1:
        warnings.warn('Expect the input mesh to be 1D, but received mesh.devices.ndim=%d. The first axis will be used for data-parallel sharding.', device_mesh.devices.ndim)