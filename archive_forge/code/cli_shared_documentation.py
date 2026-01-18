import math
import numpy as np
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import tensor_format
from tensorflow.python.debug.lib import common
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
Generate formatted intro for TensorFlow run-time error.

  Args:
    tf_error: (errors.OpError) TensorFlow run-time error object.

  Returns:
    (RichTextLines) Formatted intro message about the run-time OpError, with
      sample commands for debugging.
  