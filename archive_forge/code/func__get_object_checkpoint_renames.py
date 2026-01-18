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
def _get_object_checkpoint_renames(path, variable_names):
    """Returns a dictionary mapping variable names to checkpoint keys.

  The warm-starting utility expects variable names to match with the variable
  names in the checkpoint. For object-based checkpoints, the variable names
  and names in the checkpoint are different. Thus, for object-based checkpoints,
  this function is used to obtain the map from variable names to checkpoint
  keys.

  Args:
    path: path to checkpoint directory or file.
    variable_names: list of variable names to load from the checkpoint.

  Returns:
    If the checkpoint is object-based, this function returns a map from variable
    names to their corresponding checkpoint keys.
    If the checkpoint is name-based, this returns an empty dict.

  Raises:
    ValueError: If the object-based checkpoint is missing variables.
  """
    fname = checkpoint_utils._get_checkpoint_filename(path)
    try:
        names_to_keys = saver_lib.object_graph_key_mapping(fname)
    except errors.NotFoundError:
        return {}
    missing_names = set(variable_names) - set(names_to_keys.keys())
    if missing_names:
        raise ValueError('Attempting to warm-start from an object-based checkpoint, but found that the checkpoint did not contain values for all variables. The following variables were missing: {}'.format(missing_names))
    return {name: names_to_keys[name] for name in variable_names}