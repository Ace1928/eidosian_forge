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
def _histogram_fixed_width_eager_fallback(values: _atypes.TensorFuzzingAnnotation[TV_HistogramFixedWidth_T], value_range: _atypes.TensorFuzzingAnnotation[TV_HistogramFixedWidth_T], nbins: _atypes.TensorFuzzingAnnotation[_atypes.Int32], dtype: TV_HistogramFixedWidth_dtype, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_HistogramFixedWidth_dtype]:
    if dtype is None:
        dtype = _dtypes.int32
    dtype = _execute.make_type(dtype, 'dtype')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([values, value_range], ctx, [_dtypes.int32, _dtypes.int64, _dtypes.float32, _dtypes.float64])
    values, value_range = _inputs_T
    nbins = _ops.convert_to_tensor(nbins, _dtypes.int32)
    _inputs_flat = [values, value_range, nbins]
    _attrs = ('T', _attr_T, 'dtype', dtype)
    _result = _execute.execute(b'HistogramFixedWidth', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('HistogramFixedWidth', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result