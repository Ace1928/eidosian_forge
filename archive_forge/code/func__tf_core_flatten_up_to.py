import collections as _collections
import enum
import typing
from typing import Protocol
import six as _six
import wrapt as _wrapt
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util.compat import collections_abc as _collections_abc
def _tf_core_flatten_up_to(shallow_tree, input_tree, check_types=True, expand_composites=False):
    is_nested_fn = _is_nested_or_composite if expand_composites else _tf_core_is_nested
    _tf_core_assert_shallow_structure(shallow_tree, input_tree, check_types=check_types, expand_composites=expand_composites)
    return [v for _, v in _tf_core_yield_flat_up_to(shallow_tree, input_tree, is_nested_fn)]