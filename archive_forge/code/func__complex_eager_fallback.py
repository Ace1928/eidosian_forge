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
def _complex_eager_fallback(real: _atypes.TensorFuzzingAnnotation[TV_Complex_T], imag: _atypes.TensorFuzzingAnnotation[TV_Complex_T], Tout: TV_Complex_Tout, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_Complex_Tout]:
    if Tout is None:
        Tout = _dtypes.complex64
    Tout = _execute.make_type(Tout, 'Tout')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([real, imag], ctx, [_dtypes.float32, _dtypes.float64], _dtypes.float32)
    real, imag = _inputs_T
    _inputs_flat = [real, imag]
    _attrs = ('T', _attr_T, 'Tout', Tout)
    _result = _execute.execute(b'Complex', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('Complex', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result