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
def _update_docstring_with_api_list(target, api_list):
    """Replaces `<<API_LIST>>` in target.__doc__ with the given list of APIs."""
    lines = []
    for func in api_list:
        name = tf_export_lib.get_canonical_name_for_symbol(func, add_prefix_to_v1_names=True)
        if name is not None:
            params = tf_inspect.signature(func).parameters.keys()
            lines.append(f'  * `tf.{name}({', '.join(params)})`')
    lines.sort()
    target.__doc__ = target.__doc__.replace('  <<API_LIST>>', '\n'.join(lines))