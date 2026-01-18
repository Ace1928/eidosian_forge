import os
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.lib.io import file_io
Removes the temp path for file after writing is finished.

  Args:
    filepath: Original filepath that would be used without distribution.
    strategy: The tf.distribute strategy object currently used.
  