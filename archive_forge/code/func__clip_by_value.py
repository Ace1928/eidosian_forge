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
def _clip_by_value(t: _atypes.TensorFuzzingAnnotation[TV_ClipByValue_T], clip_value_min: _atypes.TensorFuzzingAnnotation[TV_ClipByValue_T], clip_value_max: _atypes.TensorFuzzingAnnotation[TV_ClipByValue_T], name=None) -> _atypes.TensorFuzzingAnnotation[TV_ClipByValue_T]:
    """Clips tensor values to a specified min and max.

  Given a tensor `t`, this operation returns a tensor of the same type and
  shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
  Any values less than `clip_value_min` are set to `clip_value_min`. Any values
  greater than `clip_value_max` are set to `clip_value_max`.

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A `Tensor`.
    clip_value_min: A `Tensor`. Must have the same type as `t`.
      A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
      as `t`. The minimum value to clip by.
    clip_value_max: A `Tensor`. Must have the same type as `t`.
      A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
      as `t`. The maximum value to clip by.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ClipByValue', name, t, clip_value_min, clip_value_max)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return _clip_by_value_eager_fallback(t, clip_value_min, clip_value_max, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ClipByValue', t=t, clip_value_min=clip_value_min, clip_value_max=clip_value_max, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ClipByValue', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result