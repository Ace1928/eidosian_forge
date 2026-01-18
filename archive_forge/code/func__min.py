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
def _min(input: _atypes.TensorFuzzingAnnotation[TV_Min_T], axis: _atypes.TensorFuzzingAnnotation[TV_Min_Tidx], keep_dims: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_Min_T]:
    """Computes the minimum of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Min', name, input, axis, 'keep_dims', keep_dims)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return _min_eager_fallback(input, axis, keep_dims=keep_dims, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if keep_dims is None:
        keep_dims = False
    keep_dims = _execute.make_bool(keep_dims, 'keep_dims')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('Min', input=input, reduction_indices=axis, keep_dims=keep_dims, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('keep_dims', _op._get_attr_bool('keep_dims'), 'T', _op._get_attr_type('T'), 'Tidx', _op._get_attr_type('Tidx'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Min', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result