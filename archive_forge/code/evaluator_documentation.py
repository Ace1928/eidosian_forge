import re
import numpy as np  # pylint: disable=unused-import
from tensorflow.python.debug.lib import debug_data
Parse an expression.

    Args:
      expression: the expression to be parsed.

    Returns:
      The result of the evaluation.

    Raises:
      ValueError: If the value of one or more of the debug tensors in the
        expression are not available.
    