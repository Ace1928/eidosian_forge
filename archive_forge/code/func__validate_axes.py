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
def _validate_axes(self):
    if self._device_mesh:
        valid_axis_names = set(self._device_mesh.axis_names)
        axis_names = set(self._axes) - set([None])
        if axis_names - valid_axis_names:
            raise ValueError(f'Invalid axis names for Layout. Valid axis names: {valid_axis_names}, Got {axis_names}')