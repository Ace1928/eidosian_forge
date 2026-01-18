import sys
import textwrap
from typing import Union
import numpy as np
from tensorflow.python.types import doc_typealias
from tensorflow.python.util.tf_export import tf_export
@runtime_checkable
class TensorProtocol(Protocol):
    """Protocol type for objects that can be converted to Tensor."""

    def __tf_tensor__(self, dtype=None, name=None):
        """Converts this object to a Tensor.

    Args:
      dtype: data type for the returned Tensor
      name: a name for the operations which create the Tensor
    Returns:
      A Tensor.
    """
        pass