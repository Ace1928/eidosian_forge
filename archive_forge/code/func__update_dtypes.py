import base64
import collections
import logging
import os
from .. import prediction_utils
from .._interfaces import PredictionClient
import numpy as np
from ..prediction_utils import PredictionError
import six
import tensorflow as tf
def _update_dtypes(graph, interface):
    """Adds dtype to TensorInfos in interface if necessary.

  If already present, validates TensorInfo matches values in the graph.
  TensorInfo is updated in place.

  Args:
    graph: the TensorFlow graph; used to lookup datatypes of tensors.
    interface: map from alias to TensorInfo object.

  Raises:
    ValueError: if the data type in the TensorInfo does not match the type
      found in graph.
  """
    for alias, info in six.iteritems(interface):
        dtype = graph.get_tensor_by_name(info.name).dtype
        if not info.dtype:
            info.dtype = dtype.as_datatype_enum
        elif info.dtype != dtype.as_datatype_enum:
            raise ValueError('Specified data types do not match for alias %s. Graph has %d while TensorInfo reports %d.' % (alias, dtype, info.dtype))