import abc
import functools
import queue
import threading
import warnings
import numpy as np
from tensorflow.core.framework import dataset_metadata_pb2
from tensorflow.core.framework import dataset_options_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_autograph
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.data.util import traverse
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd_utils
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as core_random_seed
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base as tracking_base
from tensorflow.python.trackable import resource as resource_lib
from tensorflow.python.types import data as data_types
from tensorflow.python.types import trace
from tensorflow.python.util import deprecation
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import nest as tf_nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('data.DatasetSpec', v1=['data.DatasetSpec', 'data.experimental.DatasetStructure'])
class DatasetSpec(type_spec.BatchableTypeSpec):
    """Type specification for `tf.data.Dataset`.

  See `tf.TypeSpec` for more information about TensorFlow type specifications.

  >>> dataset = tf.data.Dataset.range(3)
  >>> tf.data.DatasetSpec.from_value(dataset)
  DatasetSpec(TensorSpec(shape=(), dtype=tf.int64, name=None), TensorShape([]))
  """
    __slots__ = ['_element_spec', '_dataset_shape']

    def __init__(self, element_spec, dataset_shape=()):
        self._element_spec = element_spec
        self._dataset_shape = tensor_shape.as_shape(dataset_shape)

    @property
    def value_type(self):
        return Dataset

    @property
    def element_spec(self):
        """The inner element spec."""
        return self._element_spec

    def is_subtype_of(self, other):
        """See base class."""
        if type(self) is not type(other):
            return False
        try:
            tf_nest.assert_same_structure(self.element_spec, other.element_spec)
        except (TypeError, ValueError):
            return False
        self_elements = tf_nest.flatten(self.element_spec)
        other_elements = tf_nest.flatten(other.element_spec)

        def is_subtype_or_equal(a, b):
            if isinstance(a, trace.TraceType):
                return a.is_subtype_of(b)
            else:
                return a == b
        for self_element, other_element in zip(self_elements, other_elements):
            if not is_subtype_or_equal(self_element, other_element):
                return False
        return self._dataset_shape.is_subtype_of(other._dataset_shape)

    def most_specific_common_supertype(self, others):
        """See base class."""
        if not all((type(self) is type(other) for other in others)):
            return None
        try:
            for other in others:
                tf_nest.assert_same_structure(self.element_spec, other.element_spec)
        except (TypeError, ValueError):
            return None
        self_components = tf_nest.flatten(self.element_spec)
        others_components = [tf_nest.flatten(other.element_spec) for other in others]
        common_components = [None] * len(self_components)

        def common_supertype_or_equal(a, bs):
            if isinstance(a, trace.TraceType):
                return a.most_specific_common_supertype(bs)
            else:
                return a if all((a == b for b in bs)) else None
        for i, self_component in enumerate(self_components):
            common_components[i] = common_supertype_or_equal(self_component, [other_components[i] for other_components in others_components])
            if self_component is not None and common_components[i] is None:
                return None
        common_element_spec = tf_nest.pack_sequence_as(self._element_spec, common_components)
        common_dataset_shape = self._dataset_shape.most_specific_common_supertype([other._dataset_shape for other in others])
        if common_dataset_shape is None:
            return None
        return DatasetSpec(common_element_spec, common_dataset_shape)

    def _serialize(self):
        return (self._element_spec, self._dataset_shape)

    @property
    def _component_specs(self):
        return tensor_spec.TensorSpec(self._dataset_shape, dtypes.variant)

    def _to_components(self, value):
        return value._variant_tensor

    def _from_components(self, components):
        if self._dataset_shape.ndims == 0:
            return _VariantDataset(components, self._element_spec)
        else:
            return _NestedVariant(components, self._element_spec, self._dataset_shape)

    def _to_tensor_list(self, value):
        return [ops.convert_to_tensor(tf_nest.map_structure(lambda x: x._variant_tensor, value))]

    @staticmethod
    def from_value(value):
        """Creates a `DatasetSpec` for the given `tf.data.Dataset` value."""
        return DatasetSpec(value.element_spec)

    def _batch(self, batch_size):
        return DatasetSpec(self._element_spec, tensor_shape.TensorShape([batch_size]).concatenate(self._dataset_shape))

    def _unbatch(self):
        if self._dataset_shape.ndims == 0:
            raise ValueError('Slicing dataset elements is not supported for rank 0.')
        return DatasetSpec(self._element_spec, self._dataset_shape[1:])

    def _to_batched_tensor_list(self, value):
        if self._dataset_shape.ndims == 0:
            raise ValueError('Slicing dataset elements is not supported for rank 0.')
        return self._to_tensor_list(value)

    def _to_legacy_output_types(self):
        return self

    def _to_legacy_output_shapes(self):
        return self

    def _to_legacy_output_classes(self):
        return self

    def __hash__(self):
        return hash(DatasetSpec)

    def __eq__(self, other):
        return isinstance(other, DatasetSpec) and self._element_spec == other._element_spec and (self._dataset_shape == other._dataset_shape)