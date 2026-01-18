import sys
import textwrap
from typing import Union
import numpy as np
from tensorflow.python.types import doc_typealias
from tensorflow.python.util.tf_export import tf_export
class Value(Tensor):
    """Tensor that can be associated with a value (aka "eager tensor").

  These objects represent the (usually future) output of executing an op
  immediately.
  """

    def numpy(self):
        pass