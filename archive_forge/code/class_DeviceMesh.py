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
@keras_export('keras.distribution.DeviceMesh')
class DeviceMesh:
    """A cluster of computation devices for distributed computation.

    This API is aligned with `jax.sharding.Mesh` and `tf.dtensor.Mesh`, which
    represents the computation devices in the global context.

    See more details in [jax.sharding.Mesh](
        https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Mesh)
    and [tf.dtensor.Mesh](
        https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Mesh).

    Args:
        shape: tuple of list of integers. The shape of the overall
            `DeviceMesh`, e.g. `(8,)` for a data parallel only distribution,
            or `(4, 2)` for a model+data parallel distribution.
        axis_names: List of string. The logical name of the each axis for
            the `DeviceMesh`. The length of the `axis_names` should match to
            the rank of the `shape`. The `axis_names` will be used to
            match/create the `TensorLayout` when distribute the data and
            variables.
        devices: Optional list of devices. Defaults to all the available
            devices locally from `keras.distribution.list_devices()`.
    """

    def __init__(self, shape, axis_names, devices=None):
        if not shape or not axis_names:
            raise ValueError(f'Shape and axis_names cannot be empty. Received: shape={shape}, axis_names={axis_names}')
        if len(shape) != len(axis_names):
            raise ValueError(f'Shape and axis_names should have same size. Received: shape={shape}, axis_names={axis_names}')
        if devices is None:
            devices = list_devices()
        devices = np.array(devices)
        if np.prod(shape) != np.prod(devices.shape):
            raise ValueError(f'Shape does not match the number of devices. Received: shape={shape}; devices.shape={devices.shape}')
        self._shape = shape
        self._axis_names = axis_names
        self._devices = np.reshape(devices, shape)

    @property
    def shape(self):
        return self._shape

    @property
    def axis_names(self):
        return self._axis_names

    @property
    def devices(self):
        return self._devices

    def __repr__(self):
        return f'<{self.__class__.__name__} shape={self.shape}, axis_names={self.axis_names}>'

    def __str__(self):
        return self.__repr__()