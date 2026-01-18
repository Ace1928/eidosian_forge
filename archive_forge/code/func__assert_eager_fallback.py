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
def _assert_eager_fallback(condition: _atypes.TensorFuzzingAnnotation[_atypes.Bool], data, summarize: int, name, ctx):
    if summarize is None:
        summarize = 3
    summarize = _execute.make_int(summarize, 'summarize')
    _attr_T, data = _execute.convert_to_mixed_eager_tensors(data, ctx)
    condition = _ops.convert_to_tensor(condition, _dtypes.bool)
    _inputs_flat = [condition] + list(data)
    _attrs = ('T', _attr_T, 'summarize', summarize)
    _result = _execute.execute(b'Assert', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result