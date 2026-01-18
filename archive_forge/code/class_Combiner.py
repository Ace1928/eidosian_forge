import abc
import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as trackable
class Combiner(object):
    """Functional object that defines a shardable computation.

  This object defines functions required to create and manipulate data objects.
  These data objects, referred to below as 'accumulators', are computation-
  specific and may be implemented alongside concrete subclasses of Combiner
  (if necessary - some computations may be simple enough that standard Python
  types can be used as accumulators).

  The intent for this class is that by describing computations in this way, we
  can arbitrarily shard a dataset, perform computations on a subset, and then
  merge the computation into a final result. This enables distributed
  computation.

  The combiner itself does not own any state - all computational state is owned
  by the accumulator objects. This is so that we can have an arbitrary number of
  Combiners (thus sharding the computation N ways) without risking any change
  to the underlying computation. These accumulator objects are uniquely
  associated with each Combiner; a Combiner defines what the accumulator object
  should be and will only work with accumulators of that type.
  """
    __metaclass__ = abc.ABCMeta

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    @abc.abstractmethod
    def compute(self, batch_values, accumulator=None):
        """Compute a step in this computation, returning a new accumulator.

    This method computes a step of the computation described by this Combiner.
    If an accumulator is passed, the data in that accumulator is also used; so
    compute(batch_values) results in f(batch_values), while
    compute(batch_values, accumulator) results in
    merge(f(batch_values), accumulator).

    Args:
      batch_values: A list of ndarrays representing the values of the inputs for
        this step of the computation.
      accumulator: the current accumulator. Can be None.

    Returns:
      An accumulator that includes the passed batch of inputs.
    """
        pass

    @abc.abstractmethod
    def merge(self, accumulators):
        """Merge several accumulators to a single accumulator.

    This method takes the partial values in several accumulators and combines
    them into a single accumulator. This computation must not be order-specific
    (that is, merge([a, b]) must return the same result as merge([b, a]).

    Args:
      accumulators: the accumulators to merge, as a list.

    Returns:
      A merged accumulator.
    """
        pass

    @abc.abstractmethod
    def extract(self, accumulator):
        """Convert an accumulator into a dict of output values.

    Args:
      accumulator: The accumulator to convert.

    Returns:
      A dict of ndarrays representing the data in this accumulator.
    """
        pass

    @abc.abstractmethod
    def restore(self, output):
        """Create an accumulator based on 'output'.

    This method creates a new accumulator with identical internal state to the
    one used to create the data in 'output'. This means that if you do

    output_data = combiner.extract(accumulator_1)
    accumulator_2 = combiner.restore(output_data)

    then accumulator_1 and accumulator_2 will have identical internal state, and
    computations using either of them will be equivalent.

    Args:
      output: The data output from a previous computation. Should be in the same
        form as provided by 'extract_output'.

    Returns:
      A new accumulator.
    """
        pass

    @abc.abstractmethod
    def serialize(self, accumulator):
        """Serialize an accumulator for a remote call.

    This function serializes an accumulator to be sent to a remote process.

    Args:
      accumulator: The accumulator to serialize.

    Returns:
      A byte string representing the passed accumulator.
    """
        pass

    @abc.abstractmethod
    def deserialize(self, encoded_accumulator):
        """Deserialize an accumulator received from 'serialize()'.

    This function deserializes an accumulator serialized by 'serialize()'.

    Args:
      encoded_accumulator: A byte string representing an accumulator.

    Returns:
      The accumulator represented by the passed byte_string.
    """
        pass