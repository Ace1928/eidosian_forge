import inspect
import numbers
import os
import re
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
def _np_doc_helper(f, np_f, np_fun_name=None, unsupported_params=None, link=None):
    """Helper to get docs."""
    assert np_f or np_fun_name
    if not np_fun_name:
        np_fun_name = np_f.__name__
    doc = "TensorFlow variant of NumPy's `%s`.\n\n" % np_fun_name
    if unsupported_params:
        doc += 'Unsupported arguments: ' + ', '.join(('`' + name + '`' for name in unsupported_params)) + '.\n\n'
    if _has_docstring(f):
        doc += f.__doc__
        doc = _add_blank_line(doc)
    doc = _add_np_doc(doc, np_fun_name, np_f, link=link)
    return doc