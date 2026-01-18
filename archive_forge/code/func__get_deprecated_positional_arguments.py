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
def _get_deprecated_positional_arguments(names_to_ok_vals, arg_spec):
    """Builds a dictionary from deprecated arguments to their spec.

    Returned dict is keyed by argument name.
    Each value is a DeprecatedArgSpec with the following fields:
       position: The zero-based argument position of the argument
         within the signature.  None if the argument isn't found in
         the signature.
       ok_values:  Values of this argument for which warning will be
         suppressed.

    Args:
      names_to_ok_vals: dict from string arg_name to a list of values, possibly
        empty, which should not elicit a warning.
      arg_spec: Output from tf_inspect.getfullargspec on the called function.

    Returns:
      Dictionary from arg_name to DeprecatedArgSpec.
    """
    arg_space = arg_spec.args + arg_spec.kwonlyargs
    arg_name_to_pos = {name: pos for pos, name in enumerate(arg_space)}
    deprecated_positional_args = {}
    for arg_name, spec in iter(names_to_ok_vals.items()):
        if arg_name in arg_name_to_pos:
            pos = arg_name_to_pos[arg_name]
            deprecated_positional_args[arg_name] = DeprecatedArgSpec(pos, spec.has_ok_value, spec.ok_value)
    return deprecated_positional_args