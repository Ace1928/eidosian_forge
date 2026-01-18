import functools
import threading
import traceback  # pylint: disable=unused-import
import weakref
import numpy as np
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.lib.core import _pywrap_py_func
from tensorflow.python.ops import autograph_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_script_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _check_args_and_maybe_make_decorator(script_op, script_op_name, func=None, inp=None, Tout=None, **kwargs):
    """Checks the arguments and returns a decorator if func is None."""
    if Tout is None:
        raise TypeError(f"Missing required argument: 'Tout'\n  If using {script_op_name} as a decorator, set `Tout`\n  **by name** above the function:\n  `@{script_op_name}(Tout=tout)`")
    if func is None:
        if inp is not None:
            raise TypeError(f"Don't set the `inp` argument when using {script_op_name} as a decorator (`func=None`).")

        def py_function_decorator(fun):

            @functools.wraps(fun)
            def py_function_wrapper(*args):
                return script_op(fun, inp=args, Tout=Tout, **kwargs)
            return py_function_wrapper
        return py_function_decorator
    if inp is None:
        raise TypeError(f'Missing argument `inp`:\n  You must set the `inp` argument (the list of arguments to the\n  function), unless you use `{script_op_name}` as a decorator(`func=None`).')
    return None