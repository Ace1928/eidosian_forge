import collections
import functools
import inspect
import re
from tensorflow.python.framework import strict_mode
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import doc_controls
def _same_value(a, b):
    """A comparison operation that works for multiple object types.

      Returns True for two empty lists, two numeric values with the
      same value, etc.

      Returns False for (pd.DataFrame, None), and other pairs which
      should not be considered equivalent.

      Args:
        a: value one of the comparison.
        b: value two of the comparison.

      Returns:
        A boolean indicating whether the two inputs are the same value
        for the purposes of deprecation.
      """
    if a is b:
        return True
    try:
        equality = a == b
        if isinstance(equality, bool):
            return equality
    except TypeError:
        return False
    return False