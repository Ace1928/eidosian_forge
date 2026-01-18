import abc
import collections
import math
import numpy as np
import six
from tensorflow.python.eager import context
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@six.add_metaclass(abc.ABCMeta)
class _FeatureColumn(object):
    """Represents a feature column abstraction.

  WARNING: Do not subclass this layer unless you know what you are doing:
  the API is subject to future changes.

  To distinguish the concept of a feature family and a specific binary feature
  within a family, we refer to a feature family like "country" as a feature
  column. Following is an example feature in a `tf.Example` format:
    {key: "country",  value: [ "US" ]}
  In this example the value of feature is "US" and "country" refers to the
  column of the feature.

  This class is an abstract class. User should not create instances of this.
  """

    @abc.abstractproperty
    def name(self):
        """Returns string. Used for naming and for name_scope."""
        pass

    def __lt__(self, other):
        """Allows feature columns to be sorted in Python 3 as they are in Python 2.

    Feature columns need to occasionally be sortable, for example when used as
    keys in a features dictionary passed to a layer.

    In CPython, `__lt__` must be defined for all objects in the
    sequence being sorted. If any objects do not have an `__lt__` compatible
    with feature column objects (such as strings), then CPython will fall back
    to using the `__gt__` method below.
    https://docs.python.org/3/library/stdtypes.html#list.sort

    Args:
      other: The other object to compare to.

    Returns:
      True if the string representation of this object is lexicographically less
      than the string representation of `other`. For FeatureColumn objects,
      this looks like "<__main__.FeatureColumn object at 0xa>".
    """
        return str(self) < str(other)

    def __gt__(self, other):
        """Allows feature columns to be sorted in Python 3 as they are in Python 2.

    Feature columns need to occasionally be sortable, for example when used as
    keys in a features dictionary passed to a layer.

    `__gt__` is called when the "other" object being compared during the sort
    does not have `__lt__` defined.
    Example:
    ```
    # __lt__ only class
    class A():
      def __lt__(self, other): return str(self) < str(other)

    a = A()
    a < "b" # True
    "0" < a # Error

    # __lt__ and __gt__ class
    class B():
      def __lt__(self, other): return str(self) < str(other)
      def __gt__(self, other): return str(self) > str(other)

    b = B()
    b < "c" # True
    "0" < b # True
    ```


    Args:
      other: The other object to compare to.

    Returns:
      True if the string representation of this object is lexicographically
      greater than the string representation of `other`. For FeatureColumn
      objects, this looks like "<__main__.FeatureColumn object at 0xa>".
    """
        return str(self) > str(other)

    @property
    def _var_scope_name(self):
        """Returns string. Used for variable_scope. Defaults to self.name."""
        return self.name

    @abc.abstractmethod
    def _transform_feature(self, inputs):
        """Returns intermediate representation (usually a `Tensor`).

    Uses `inputs` to create an intermediate representation (usually a `Tensor`)
    that other feature columns can use.

    Example usage of `inputs`:
    Let's say a Feature column depends on raw feature ('raw') and another
    `_FeatureColumn` (input_fc). To access corresponding `Tensor`s, inputs will
    be used as follows:

    ```python
    raw_tensor = inputs.get('raw')
    fc_tensor = inputs.get(input_fc)
    ```

    Args:
      inputs: A `_LazyBuilder` object to access inputs.

    Returns:
      Transformed feature `Tensor`.
    """
        pass

    @abc.abstractproperty
    def _parse_example_spec(self):
        """Returns a `tf.Example` parsing spec as dict.

    It is used for get_parsing_spec for `tf.io.parse_example`. Returned spec is
    a dict from keys ('string') to `VarLenFeature`, `FixedLenFeature`, and other
    supported objects. Please check documentation of `tf.io.parse_example` for
    all supported spec objects.

    Let's say a Feature column depends on raw feature ('raw') and another
    `_FeatureColumn` (input_fc). One possible implementation of
    _parse_example_spec is as follows:

    ```python
    spec = {'raw': tf.io.FixedLenFeature(...)}
    spec.update(input_fc._parse_example_spec)
    return spec
    ```
    """
        pass

    def _reset_config(self):
        """Resets the configuration in the column.

    Some feature columns e.g. embedding or shared embedding columns might
    have some state that is needed to be reset sometimes. Use this method
    in that scenario.
    """