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
def _complex(real: _atypes.TensorFuzzingAnnotation[TV_Complex_T], imag: _atypes.TensorFuzzingAnnotation[TV_Complex_T], Tout: TV_Complex_Tout=_dtypes.complex64, name=None) -> _atypes.TensorFuzzingAnnotation[TV_Complex_Tout]:
    """Converts two real numbers to a complex number.

  Given a tensor `real` representing the real part of a complex number, and a
  tensor `imag` representing the imaginary part of a complex number, this
  operation returns complex numbers elementwise of the form \\\\(a + bj\\\\), where
  *a* represents the `real` part and *b* represents the `imag` part.

  The input tensors `real` and `imag` must have the same shape.

  For example:

  ```
  # tensor 'real' is [2.25, 3.25]
  # tensor `imag` is [4.75, 5.75]
  tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
  ```

  Args:
    real: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    imag: A `Tensor`. Must have the same type as `real`.
    Tout: An optional `tf.DType` from: `tf.complex64, tf.complex128`. Defaults to `tf.complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Complex', name, real, imag, 'Tout', Tout)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return _complex_eager_fallback(real, imag, Tout=Tout, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if Tout is None:
        Tout = _dtypes.complex64
    Tout = _execute.make_type(Tout, 'Tout')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('Complex', real=real, imag=imag, Tout=Tout, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tout', _op._get_attr_type('Tout'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Complex', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result