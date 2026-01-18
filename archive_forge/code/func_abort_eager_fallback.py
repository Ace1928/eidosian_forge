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
def abort_eager_fallback(error_msg: str, exit_without_error: bool, name, ctx):
    if error_msg is None:
        error_msg = ''
    error_msg = _execute.make_str(error_msg, 'error_msg')
    if exit_without_error is None:
        exit_without_error = False
    exit_without_error = _execute.make_bool(exit_without_error, 'exit_without_error')
    _inputs_flat = []
    _attrs = ('error_msg', error_msg, 'exit_without_error', exit_without_error)
    _result = _execute.execute(b'Abort', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result