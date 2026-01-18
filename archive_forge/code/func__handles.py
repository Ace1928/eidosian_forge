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
def _handles(self, args, kwargs):
    for arg in itertools.chain(args, kwargs.values()):
        if isinstance(arg, self._types) or (isinstance(arg, (list, tuple)) and any((isinstance(elt, self._types) for elt in arg))):
            return True
    return False