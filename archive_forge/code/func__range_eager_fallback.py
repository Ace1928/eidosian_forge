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
def _range_eager_fallback(start: _atypes.TensorFuzzingAnnotation[TV_Range_Tidx], limit: _atypes.TensorFuzzingAnnotation[TV_Range_Tidx], delta: _atypes.TensorFuzzingAnnotation[TV_Range_Tidx], name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_Range_Tidx]:
    _attr_Tidx, _inputs_Tidx = _execute.args_to_matching_eager([start, limit, delta], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint16, _dtypes.uint32], _dtypes.int32)
    start, limit, delta = _inputs_Tidx
    _inputs_flat = [start, limit, delta]
    _attrs = ('Tidx', _attr_Tidx)
    _result = _execute.execute(b'Range', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('Range', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result