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
def _extract_name_arg(args, kwargs, name_index):
    """Extracts the parameter `name` and returns `(args, kwargs, name_value)`."""
    if name_index < 0:
        name_value = None
    elif name_index < len(args):
        name_value = args[name_index]
        args = args[:name_index] + args[name_index + 1:]
    else:
        name_value = kwargs.pop('name', None)
    return (args, kwargs, name_value)