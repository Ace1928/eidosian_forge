import abc
import contextlib
import types
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _update_docstring(old_str, append_str):
    """Update old_str by inserting append_str just before the "Args:" section."""
    old_str = old_str or ''
    old_str_lines = old_str.split('\n')
    append_str = '\n'.join(('    %s' % line for line in append_str.split('\n')))
    has_args_ix = [ix for ix, line in enumerate(old_str_lines) if line.strip().lower() == 'args:']
    if has_args_ix:
        final_args_ix = has_args_ix[-1]
        return '\n'.join(old_str_lines[:final_args_ix]) + '\n\n' + append_str + '\n\n' + '\n'.join(old_str_lines[final_args_ix:])
    else:
        return old_str + '\n\n' + append_str