from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_export
Calls this decorator.

    Args:
      func: decorated symbol (function or class).

    Returns:
      The input function with _tf_api_names attribute set and marked as
      deprecated.
    