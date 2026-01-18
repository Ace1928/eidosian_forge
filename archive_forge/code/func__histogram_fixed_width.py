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
def _histogram_fixed_width(values: _atypes.TensorFuzzingAnnotation[TV_HistogramFixedWidth_T], value_range: _atypes.TensorFuzzingAnnotation[TV_HistogramFixedWidth_T], nbins: _atypes.TensorFuzzingAnnotation[_atypes.Int32], dtype: TV_HistogramFixedWidth_dtype=_dtypes.int32, name=None) -> _atypes.TensorFuzzingAnnotation[TV_HistogramFixedWidth_dtype]:
    """Return histogram of values.

  Given the tensor `values`, this operation returns a rank 1 histogram counting
  the number of entries in `values` that fall into every bin.  The bins are
  equal width and determined by the arguments `value_range` and `nbins`.

  ```python
  # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
  nbins = 5
  value_range = [0.0, 5.0]
  new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]

  with tf.get_default_session() as sess:
    hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
    variables.global_variables_initializer().run()
    sess.run(hist) => [2, 1, 1, 0, 2]
  ```

  Args:
    values: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      Numeric `Tensor`.
    value_range: A `Tensor`. Must have the same type as `values`.
      Shape [2] `Tensor` of same `dtype` as `values`.
      values <= value_range[0] will be mapped to hist[0],
      values >= value_range[1] will be mapped to hist[-1].
    nbins: A `Tensor` of type `int32`.
      Scalar `int32 Tensor`.  Number of histogram bins.
    dtype: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'HistogramFixedWidth', name, values, value_range, nbins, 'dtype', dtype)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return _histogram_fixed_width_eager_fallback(values, value_range, nbins, dtype=dtype, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if dtype is None:
        dtype = _dtypes.int32
    dtype = _execute.make_type(dtype, 'dtype')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('HistogramFixedWidth', values=values, value_range=value_range, nbins=nbins, dtype=dtype, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'dtype', _op._get_attr_type('dtype'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('HistogramFixedWidth', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result