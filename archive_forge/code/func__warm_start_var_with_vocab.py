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
def _warm_start_var_with_vocab(var, current_vocab_path, current_vocab_size, prev_ckpt, prev_vocab_path, previous_vocab_size=-1, current_oov_buckets=0, prev_tensor_name=None, initializer=None, axis=0):
    """Warm-starts given variable from `prev_tensor_name` tensor in `prev_ckpt`.

  Use this method when the `var` is backed by vocabulary. This method stitches
  the given `var` such that values corresponding to individual features in the
  vocabulary remain consistent irrespective of changing order of the features
  between old and new vocabularies.

  Args:
    var: Current graph's variable that needs to be warm-started (initialized).
      Can be either of the following:
      (i) `Variable`
      (ii) `ResourceVariable`
      (iii) list of `Variable`: The list must contain slices of the same larger
        variable.
      (iv) `PartitionedVariable`
    current_vocab_path: Path to the vocab file used for the given `var`.
    current_vocab_size: An `int` specifying the number of entries in the current
      vocab.
    prev_ckpt: A string specifying the directory with checkpoint file(s) or path
      to checkpoint. The given checkpoint must have tensor with name
      `prev_tensor_name` (if not None) or tensor with name same as given `var`.
    prev_vocab_path: Path to the vocab file used for the tensor in `prev_ckpt`.
    previous_vocab_size: If provided, will constrain previous vocab to the first
      `previous_vocab_size` entries.  -1 means use the entire previous vocab.
    current_oov_buckets: An `int` specifying the number of out-of-vocabulary
      buckets used for given `var`.
    prev_tensor_name: Name of the tensor to lookup in provided `prev_ckpt`. If
      None, we lookup tensor with same name as given `var`.
    initializer: Variable initializer to be used for missing entries.  If None,
      missing entries will be zero-initialized.
    axis: Axis of the variable that the provided vocabulary corresponds to.

  Raises:
    ValueError: If required args are not provided.
  """
    if not (current_vocab_path and current_vocab_size and prev_ckpt and prev_vocab_path):
        raise ValueError('Invalid args: Must provide all of [current_vocab_path, current_vocab_size, prev_ckpt, prev_vocab_path}.')
    if checkpoint_utils._is_variable(var):
        var = [var]
    elif isinstance(var, list) and all((checkpoint_utils._is_variable(v) for v in var)):
        var = var
    elif isinstance(var, variables_lib.PartitionedVariable):
        var = var._get_variable_list()
    else:
        raise TypeError('var MUST be one of the following: a Variable, list of Variable or PartitionedVariable, but is {}'.format(type(var)))
    if not prev_tensor_name:
        prev_tensor_name = _infer_var_name(var)
    total_v_first_axis = sum((v.get_shape().as_list()[0] for v in var))
    for v in var:
        v_shape = v.get_shape().as_list()
        slice_info = v._get_save_slice_info()
        partition_info = None
        if slice_info:
            partition_info = variable_scope._PartitionInfo(full_shape=slice_info.full_shape, var_offset=slice_info.var_offset)
        if axis == 0:
            new_row_vocab_size = current_vocab_size
            new_col_vocab_size = v_shape[1]
            old_row_vocab_size = previous_vocab_size
            old_row_vocab_file = prev_vocab_path
            new_row_vocab_file = current_vocab_path
            old_col_vocab_file = None
            new_col_vocab_file = None
            num_row_oov_buckets = current_oov_buckets
            num_col_oov_buckets = 0
        elif axis == 1:
            new_row_vocab_size = total_v_first_axis
            new_col_vocab_size = current_vocab_size
            old_row_vocab_size = -1
            old_row_vocab_file = None
            new_row_vocab_file = None
            old_col_vocab_file = prev_vocab_path
            new_col_vocab_file = current_vocab_path
            num_row_oov_buckets = 0
            num_col_oov_buckets = current_oov_buckets
        else:
            raise ValueError('The only supported values for the axis argument are 0 and 1.  Provided axis: {}'.format(axis))
        init = checkpoint_ops._load_and_remap_matrix_initializer(ckpt_path=checkpoint_utils._get_checkpoint_filename(prev_ckpt), old_tensor_name=prev_tensor_name, new_row_vocab_size=new_row_vocab_size, new_col_vocab_size=new_col_vocab_size, old_row_vocab_size=old_row_vocab_size, old_row_vocab_file=old_row_vocab_file, new_row_vocab_file=new_row_vocab_file, old_col_vocab_file=old_col_vocab_file, new_col_vocab_file=new_col_vocab_file, num_row_oov_buckets=num_row_oov_buckets, num_col_oov_buckets=num_col_oov_buckets, initializer=initializer)
        new_init_val = ops.convert_to_tensor(init(shape=v_shape, partition_info=partition_info))
        v._initializer_op = state_ops.assign(v, new_init_val)