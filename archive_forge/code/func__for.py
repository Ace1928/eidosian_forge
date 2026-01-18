import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def _for(start: _atypes.TensorFuzzingAnnotation[_atypes.Int32], limit: _atypes.TensorFuzzingAnnotation[_atypes.Int32], delta: _atypes.TensorFuzzingAnnotation[_atypes.Int32], input, body, name=None):
    """Applies a for loop.

    ```python
     output = input;
     for i in range(start, limit, delta)
       output = body(i, output);
    ```

  Args:
    start: A `Tensor` of type `int32`. The lower bound. An int32
    limit: A `Tensor` of type `int32`. The upper bound. An int32
    delta: A `Tensor` of type `int32`. The increment. An int32
    input: A list of `Tensor` objects.
      A list of input tensors whose types are T.
    body: A function decorated with @Defun.
          A function that takes a list of tensors (int32, T) and returns another
          list of tensors (T).
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'For', name, start, limit, delta, input, 'body', body)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return _for_eager_fallback(start, limit, delta, input, body=body, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('For', start=start, limit=limit, delta=delta, input=input, body=body, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op.get_attr('T'), 'body', _op.get_attr('body'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('For', _inputs_flat, _attrs, _result)
    return _result