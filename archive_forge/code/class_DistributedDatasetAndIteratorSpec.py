import functools
import sys
import time
import six
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.distribute_lib import InputReplicationMode
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import distribute as distribute_types
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
class DistributedDatasetAndIteratorSpec(type_spec.TypeSpec):
    """Common Type specification for `DistributedDataset and DistributedDatasetsFromFunction."""
    __slots__ = ['_input_workers', '_element_spec', '_strategy', '_cardinality', '_enable_get_next_as_optional', '_options', '_canonicalize_devices']

    def __init__(self, input_workers, element_spec, strategy, options, cardinality=cardinality_lib.UNKNOWN, enable_get_next_as_optional=None, replica_order=None):
        if isinstance(input_workers, tuple):
            raise NotImplementedError('DistributedIteratorSpec does not have support for deserialization.')
        else:
            self._input_workers = input_workers
            self._element_spec = element_spec
            self._strategy = strategy
            self._cardinality = cardinality
            self._enable_get_next_as_optional = enable_get_next_as_optional
            self._options = options
            if self._strategy:
                self._canonicalize_devices = getattr(self._strategy, '_canonicalize_devices', True)
            else:
                self._canonicalize_devices = True
            self._replica_order = replica_order

    def _serialize(self):
        return (self._input_workers.serialize(), self._element_spec, id(self._strategy), id(self._options))

    def _deserialize(self):
        raise ValueError(f'Deserialization is currently unsupported for {type(self)}.')

    def sanity_check_type(self, other):
        """Returns the most specific TypeSpec compatible with `self` and `other`.

    Args:
      other: A `TypeSpec`.

    Raises:
      ValueError: If there is no TypeSpec that is compatible with both `self`
        and `other`.
    """
        if type(self) is not type(other):
            raise ValueError('No TypeSpec is compatible with both %s and %s' % (self, other))
        if self._input_workers.serialize() != other._input_workers.serialize():
            raise ValueError('_input_workers is not compatible with both %s and %s' % (self, other))
        if self._strategy is not other._strategy:
            raise ValueError('tf.distribute strategy is not compatible with both %s and %s' % (self, other))

    def is_subtype_of(self, other):
        """Returns True if `self` is subtype of `other`.

    Args:
      other: A `TypeSpec`.
    """
        try:
            self.sanity_check_type(other)
            nest.assert_same_structure(self._element_spec, other._element_spec)
        except (TypeError, ValueError):
            return False
        self_elements = nest.flatten(self._element_spec)
        other_elements = nest.flatten(other._element_spec)
        return all((self_element.is_subtype_of(other_element) for self_element, other_element in zip(self_elements, other_elements)))

    def most_specific_common_supertype(self, others):
        """Returns the most specific supertype of `self` and `others`.

    Args:
      others: A Sequence of `TypeSpec`.

    Returns `None` if a supertype does not exist.
    """
        try:
            for other in others:
                self.sanity_check_type(other)
                nest.assert_same_structure(self._element_spec, other._element_spec)
        except (TypeError, ValueError):
            return None
        self_elements = nest.flatten(self._element_spec)
        others_elements = [nest.flatten(other._element_spec) for other in others]
        common_elements = [None] * len(self_elements)
        for i, self_element in enumerate(self_elements):
            common_elements[i] = self_element.most_specific_common_supertype([other_elements[i] for other_elements in others_elements])
            if common_elements[i] is None:
                return None
        common_element_spec = nest.pack_sequence_as(self._element_spec, common_elements)
        return type(self)(self._input_workers, common_element_spec, self._strategy, self._options, cardinality=self._cardinality, enable_get_next_as_optional=self._enable_get_next_as_optional)

    def _with_tensor_ranks_only(self):
        element_spec = nest.map_structure(lambda s: s._with_tensor_ranks_only(), self._element_spec)
        return type(self)(self._input_workers, element_spec, self._strategy, self._options, cardinality=self._cardinality, enable_get_next_as_optional=self._enable_get_next_as_optional)

    def _without_tensor_names(self):
        element_spec = nest.map_structure(lambda s: s._without_tensor_names(), self._element_spec)
        return type(self)(self._input_workers, element_spec, self._strategy, self._options, cardinality=self._cardinality, enable_get_next_as_optional=self._enable_get_next_as_optional)