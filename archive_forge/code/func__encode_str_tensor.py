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
def _encode_str_tensor(data, tensor_name):
    """Encodes tensor data of type string.

  Data is a bytes in python 3 and a string in python 2. Base 64 encode the data
  if the tensorname ends in '_bytes', otherwise convert data to a string.

  Args:
    data: Data of the tensor, type bytes in python 3, string in python 2.
    tensor_name: The corresponding name of the tensor.

  Returns:
    JSON-friendly encoded version of the data.
  """
    if isinstance(data, list):
        return [_encode_str_tensor(val, tensor_name) for val in data]
    if tensor_name.endswith('_bytes'):
        return {'b64': compat.as_text(base64.b64encode(data))}
    else:
        return compat.as_text(data)