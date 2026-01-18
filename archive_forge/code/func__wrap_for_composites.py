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
def _wrap_for_composites(func, inp, Tout):
    """Wraps user inputs to support composite tensors for `py_function`.

  1. Flattens `inp` to a list of Tensors (by flattening any composite tensors).
  2. Creates a wrapper fuction for `func` that expects flat inputs and:
     - Packs the inputs into the input structure expected by `func`.
     - Calls `func` with the packed inputs.
     - Checks that `func`'s output matches `Tout`.
     - Flattens func`'s output to a list of Tensors (flattening any composite
       tensors).

  Args:
    func: The function to wrap (`func` argument to `py_function`).
    inp: The input arguments for func (`inp` argument to `py_function`).
    Tout: The expected output types for func (`Tout` argument to `py_function).

  Returns:
    A tuple `(func, inp, Tout, out_structure)`, where `func` is the wrapped
    function, `inp` is the flattened inputs, `Tout` is the list of expected
    dtypes for the flattened outputs, and `out_structure` is the expected
    output structure (which can be used to pack the output tensors).
  """
    in_structure = [v if isinstance(v, composite_tensor.CompositeTensor) else 1 for v in inp]
    inp = nest.flatten_up_to(in_structure, inp, expand_composites=True)
    out_structure = Tout
    Tout = [v.dtype if isinstance(v, tensor_spec.TensorSpec) else v for v in nest.flatten(Tout, expand_composites=True)]

    def wrapped_func(*flat_inp):
        structured_inp = nest.pack_sequence_as(in_structure, flat_inp, expand_composites=True)
        out = func(*structured_inp)
        if not out_structure:
            return []
        if not isinstance(out, (list, tuple)):
            out = [out]
        flat_out = []
        for elt, expected_type in zip(out, out_structure):
            if isinstance(expected_type, type_spec.TypeSpec) and (not isinstance(expected_type, tensor_spec.TensorSpec)):
                if not expected_type.is_compatible_with(elt):
                    raise ValueError(f'py_function: func={func} returned {out!r}, which did not match Tout={out_structure!r}.\nIn particular, {elt!r} is not compatible with {expected_type!r}.')
                flat_out.extend(nest.flatten(elt, expand_composites=True))
            else:
                if isinstance(elt, composite_tensor.CompositeTensor):
                    raise ValueError(f'py_function: func={func} returned {out!r}, which did not match Tout={out_structure!r}.\nIn particular, {elt!r} is not a Tensor.')
                flat_out.append(elt)
        return flat_out
    return (wrapped_func, inp, Tout, out_structure)