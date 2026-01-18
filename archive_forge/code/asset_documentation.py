import os
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.trackable import base
from tensorflow.python.util.tf_export import tf_export
Fetch the current asset path.