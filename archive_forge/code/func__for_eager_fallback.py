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
def _for_eager_fallback(start: _atypes.TensorFuzzingAnnotation[_atypes.Int32], limit: _atypes.TensorFuzzingAnnotation[_atypes.Int32], delta: _atypes.TensorFuzzingAnnotation[_atypes.Int32], input, body, name, ctx):
    _attr_T, input = _execute.convert_to_mixed_eager_tensors(input, ctx)
    start = _ops.convert_to_tensor(start, _dtypes.int32)
    limit = _ops.convert_to_tensor(limit, _dtypes.int32)
    delta = _ops.convert_to_tensor(delta, _dtypes.int32)
    _inputs_flat = [start, limit, delta] + list(input)
    _attrs = ('T', _attr_T, 'body', body)
    _result = _execute.execute(b'For', len(input), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('For', _inputs_flat, _attrs, _result)
    return _result