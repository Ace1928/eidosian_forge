import os
import os.path
import re
from absl import flags
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
def _flag_value_as_int_list(self, wanted_flag_name):
    """Returns the integer list of a TensorTracer flag.

    Args:
      wanted_flag_name: the name of the flag we are looking for.

    Returns:
      the value of the flag.
    Raises:
      RuntimeError: If supposedly deadcode is reached.
    """
    int_list = []
    found, flag_value = self.get_flag_value(wanted_flag_name)
    if found and flag_value:
        try:
            integer_values = flag_value.split(',')
            int_list = [int(int_val) for int_val in integer_values]
        except ValueError:
            logging.warning('Cannot convert %s to int for flag %s', int_list, wanted_flag_name)
    return int_list