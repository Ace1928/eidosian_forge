import collections
import itertools
import typing  # pylint: disable=unused-import (used in doctests)
from tensorflow.python.framework import _pywrap_python_api_dispatcher as _api_dispatcher
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export as tf_export_lib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import type_annotations
from tensorflow.python.util.tf_export import tf_export
def _get_required_param_names(sig):
    """Returns a list of required parameter names from a python Signature."""
    params = []
    for p in sig.parameters.values():
        if p.kind == p.VAR_POSITIONAL:
            continue
        if p.kind == p.VAR_KEYWORD:
            continue
        if p.default is not p.empty:
            continue
        params.append(p.name)
    return params