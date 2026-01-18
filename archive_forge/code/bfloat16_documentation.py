from typing import Generator, Optional, Text
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
Custom getter that forces variables to have type self.variable_type.