import os
import sys
import traceback
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
Traces argument information at compilation time.

  `trace` is useful when debugging, and it always executes during the tracing
  phase, that is, when the TF graph is constructed.

  _Example usage_

  ```python
  import tensorflow as tf

  for i in tf.range(10):
    tf.autograph.trace(i)
  # Output: <Tensor ...>
  ```

  Args:
    *args: Arguments to print to `sys.stdout`.
  