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
@keras_export('keras.distribution.TensorLayout')
class TensorLayout:
    """A layout to apply to a tensor.

    This API is aligned with `jax.sharding.NamedSharding`
    and `tf.dtensor.Layout`.

    See more details in [jax.sharding.NamedSharding](
        https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.NamedSharding)
    and [tf.dtensor.Layout](
        https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Layout).

    Args:
        axes: tuple of strings that should map to the `axis_names` in
            a `DeviceMesh`. For any dimentions that doesn't need any sharding,
            A `None` can be used a placeholder.
        device_mesh: Optional `DeviceMesh` that will be used to create
            the layout. The actual mapping of tensor to physical device
            is not known until the mesh is specified.
    """

    def __init__(self, axes, device_mesh=None):
        self._axes = tuple(axes)
        self._device_mesh = device_mesh
        self._validate_axes()

    @property
    def axes(self):
        return self._axes

    @property
    def device_mesh(self):
        return self._device_mesh

    @device_mesh.setter
    def device_mesh(self, device_mesh):
        if self._device_mesh is not None:
            raise ValueError(f'Cannot override device mesh value. Existing value is {self._device_mesh}')
        self._device_mesh = device_mesh
        self._validate_axes()

    def _validate_axes(self):
        if self._device_mesh:
            valid_axis_names = set(self._device_mesh.axis_names)
            axis_names = set(self._axes) - set([None])
            if axis_names - valid_axis_names:
                raise ValueError(f'Invalid axis names for Layout. Valid axis names: {valid_axis_names}, Got {axis_names}')

    def __repr__(self):
        return f'<{self.__class__.__name__} axes={self.axes}, device_mesh={self.device_mesh}>'

    def __str__(self):
        return self.__repr__()