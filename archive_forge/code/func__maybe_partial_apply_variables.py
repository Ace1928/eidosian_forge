import atexit
import collections
import contextlib
import copy
import functools
import weakref
from absl import logging
import numpy as np
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import tpu_replicated_variable
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver as tpu_cluster_resolver_lib
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import device_assignment as device_assignment_lib  # pylint: disable=unused-import
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_hardware_feature
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _maybe_partial_apply_variables(fn, args, kwargs):
    """Inspects arguments to partially apply any DistributedVariable.

  This avoids an automatic cast of the current variable value to tensor.

  Note that a variable may be captured implicitly with Python scope instead of
  passing it to run(), but supporting run() keeps behavior consistent
  with MirroredStrategy.

  Since positional arguments must be applied from left to right, this function
  does some tricky function inspection to move variable positional arguments
  into kwargs. As a result of this, we can't support passing Variables as *args,
  nor as args to functions which combine both explicit positional arguments and
  *args.

  Args:
    fn: The function to run, as passed to run().
    args: Positional arguments to fn, as passed to run().
    kwargs: Keyword arguments to fn, as passed to run().

  Returns:
    A tuple of the function (possibly wrapped), args, kwargs (both
    possibly filtered, with members of args possibly moved to kwargs).
    If no variables are found, this function is a noop.

  Raises:
    ValueError: If the function signature makes unsupported use of *args, or if
      too many arguments are passed.
  """

    def is_distributed_var(x):
        flat = nest.flatten(x)
        return flat and isinstance(flat[0], values.DistributedVariable)
    var_kwargs = {}
    nonvar_kwargs = {}
    if kwargs:
        var_kwargs = {k: v for k, v in kwargs.items() if is_distributed_var(v)}
    if var_kwargs:
        nonvar_kwargs = {k: v for k, v in kwargs.items() if not is_distributed_var(v)}
    positional_args = []
    index_of_star_args = None
    for i, p in enumerate(tf_inspect.signature(fn).parameters.values()):
        if i == 0 and p.name == 'self':
            continue
        if p.kind == tf_inspect.Parameter.POSITIONAL_OR_KEYWORD:
            positional_args.append(p.name)
        elif p.kind == tf_inspect.Parameter.VAR_POSITIONAL:
            index_of_star_args = i
        elif p.kind == tf_inspect.Parameter.POSITIONAL_ONLY:
            if var_kwargs or any((is_distributed_var(a) for a in args)):
                raise ValueError(f'Mixing Variables and positional-only parameters not supported by TPUStrategy. Received {len(var_kwargs)} DistributedVariables in **kwargs and {sum((is_distributed_var(a) for a in args))} in *args, expected zero for both.')
            return (fn, args, kwargs)
    star_args = []
    have_seen_var_arg = False
    for i, a in enumerate(args):
        if is_distributed_var(a):
            if index_of_star_args is not None and i >= index_of_star_args:
                raise ValueError('TPUStrategy.run() cannot handle Variables passed to *args. Either name the function argument, or capture the Variable implicitly.')
            if len(positional_args) <= i:
                raise ValueError('Too many positional arguments passed to call to TPUStrategy.run().')
            var_kwargs[positional_args[i]] = a
            have_seen_var_arg = True
        else:
            if index_of_star_args is not None and i >= index_of_star_args:
                if have_seen_var_arg:
                    raise ValueError('TPUStrategy.run() cannot handle both Variables and a mix of positional args and *args. Either remove the *args, or capture the Variable implicitly.')
                else:
                    star_args.append(a)
                    continue
            if len(positional_args) <= i:
                raise ValueError('Too many positional arguments passed to call to TPUStrategy.run().')
            nonvar_kwargs[positional_args[i]] = a
    if var_kwargs:
        return (functools.partial(fn, **var_kwargs), star_args, nonvar_kwargs)
    return (fn, args, kwargs)