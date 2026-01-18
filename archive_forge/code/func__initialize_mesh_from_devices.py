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
def _initialize_mesh_from_devices(self, devices):
    devices = np.array(devices)
    device_mesh = DeviceMesh(shape=devices.shape, axis_names=[DEFAULT_BATCH_DIM_NAME], devices=devices)
    super().__init__(device_mesh)