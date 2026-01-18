from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
def _indexed_case_verify_and_canonicalize_args(branch_fns, default, branch_index):
    """Verifies input arguments for the case function.

  Args:
    branch_fns: Dict or list of pairs of an `int` and a callable which returns a
      list of tensors.
    default: Optional callable that returns a list of tensors.
    branch_index: Optional int `Tensor`, which selects for the corresponding
      pred_fn_pair.

  Raises:
    TypeError: If `branch_fns` is not a list/dictionary.
    TypeError: If `branch_fns` is a list but does not contain 2-tuples or
               callables.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.

  Returns:
    branch_fns: validated list of callables for each branch (default last).
  """
    if not isinstance(branch_index, tensor.Tensor):
        raise TypeError("'branch_index' must be a Tensor, got {}".format(type(branch_index)))
    if not branch_index.dtype.is_integer:
        raise TypeError("'branch_index' must be an integer Tensor, got {}".format(branch_index.dtype))
    if not branch_fns:
        raise ValueError("Must provide at least one item in 'branch_fns'")
    if not isinstance(branch_fns, (list, tuple, dict)):
        raise TypeError("'branch_fns' must be a list, tuple, or dict")
    if isinstance(branch_fns, dict):
        branch_fns = branch_fns.items()
    if all((callable(fn) for fn in branch_fns)):
        branch_fns = list(enumerate(branch_fns))
    for key_fn_pair in branch_fns:
        if not isinstance(key_fn_pair, tuple) or len(key_fn_pair) != 2:
            raise TypeError(f"Each entry in 'branch_fns' must be a 2-tuple. Received {key_fn_pair}.")
        key, branch_fn = key_fn_pair
        if not isinstance(key, int):
            raise TypeError('key must be a Python `int`, got {}'.format(type(key)))
        if not callable(branch_fn):
            raise TypeError('fn for key {} must be callable.'.format(key))
    keys = [p[0] for p in branch_fns]
    if min(keys) < 0 or max(keys) >= len(keys) or len(set(keys)) != len(keys):
        raise ValueError('branch indices (keys) must form contiguous range of [0 to {}) but found {{{}}}'.format(len(keys), ','.join(map(str, sorted(keys)))))
    actions = [p[1] for p in sorted(branch_fns)]
    if default is not None:
        actions.append(default)
    return actions