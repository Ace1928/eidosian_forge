from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_shape
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect
def _ragged_op_signature(op, ragged_args, ragged_varargs=False):
    """Returns a signature for the given op, marking ragged args in bold."""
    op_name = tf_export.get_canonical_name_for_symbol(op)
    argspec = tf_inspect.getfullargspec(op)
    arg_names = argspec.args
    for pos in ragged_args:
        arg_names[pos] = '**' + arg_names[pos] + '**'
    if argspec.defaults is not None:
        for pos in range(-1, -len(argspec.defaults) - 1, -1):
            arg_names[pos] += '=`{!r}`'.format(argspec.defaults[pos])
    if argspec.varargs:
        if ragged_varargs:
            arg_names.append('***' + argspec.varargs + '**')
        else:
            arg_names.append('*' + argspec.varargs)
    if argspec.varkw:
        arg_names.append('**' + argspec.varkw)
    return '* `tf.{}`({})'.format(op_name, ', '.join(arg_names))