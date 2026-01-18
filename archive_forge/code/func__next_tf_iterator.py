import functools
import numpy as np
from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import cond
from tensorflow.python.util import nest
def _next_tf_iterator(iterator, default=py_builtins.UNSPECIFIED):
    if default is py_builtins.UNSPECIFIED:
        return next(iterator)
    opt_iterate = iterator.get_next_as_optional()
    _verify_structure_compatible('the default argument', 'the iterate', default, iterator.element_spec)
    return cond.cond(opt_iterate.has_value(), opt_iterate.get_value, lambda: default)