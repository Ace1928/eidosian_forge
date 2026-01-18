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
def _tf_data_yield_flat_up_to(shallow_tree, input_tree):
    """Yields elements `input_tree` partially flattened up to `shallow_tree`."""
    if _tf_data_is_nested(shallow_tree):
        for shallow_branch, input_branch in zip(_tf_data_yield_value(shallow_tree), _tf_data_yield_value(input_tree)):
            for input_leaf in _tf_data_yield_flat_up_to(shallow_branch, input_branch):
                yield input_leaf
    else:
        yield input_tree