import collections
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_ops
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util.tf_export import tf_export
def _get_grouped_variables(vars_to_warm_start):
    """Collects and groups (possibly partitioned) variables into a dictionary.

  The variables can be provided explicitly through vars_to_warm_start, or they
  are retrieved from collections (see below).

  Args:
    vars_to_warm_start: One of the following:

      - A regular expression (string) that captures which variables to
        warm-start (see tf.compat.v1.get_collection).  This expression will
        only consider variables in the TRAINABLE_VARIABLES collection.
      - A list of strings, each representing a full variable name to warm-start.
        These will consider variables in GLOBAL_VARIABLES collection.
      - A list of Variables to warm-start.
      - `None`, in which case all variables in TRAINABLE_VARIABLES will be used.
  Returns:
    A dictionary mapping variable names (strings) to lists of Variables.
  Raises:
    ValueError: If vars_to_warm_start is not a string, `None`, a list of
      `Variables`, or a list of strings.
  """
    if isinstance(vars_to_warm_start, str) or vars_to_warm_start is None:
        logging.info('Warm-starting variables only in TRAINABLE_VARIABLES.')
        list_of_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES, scope=vars_to_warm_start)
    elif isinstance(vars_to_warm_start, list):
        if all((isinstance(v, str) for v in vars_to_warm_start)):
            list_of_vars = []
            for v in vars_to_warm_start:
                list_of_vars += ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES, scope=v)
        elif all((checkpoint_utils._is_variable(v) for v in vars_to_warm_start)):
            list_of_vars = vars_to_warm_start
        else:
            raise ValueError('If `vars_to_warm_start` is a list, it must be all `Variable` or all `str`.  Given types are {}'.format([type(v) for v in vars_to_warm_start]))
    else:
        raise ValueError('`vars_to_warm_start must be a `list` or `str`.  Given type is {}'.format(type(vars_to_warm_start)))
    grouped_variables = {}
    for v in list_of_vars:
        t = [v] if not isinstance(v, list) else v
        var_name = _infer_var_name(t)
        grouped_variables.setdefault(var_name, []).append(v)
    return grouped_variables