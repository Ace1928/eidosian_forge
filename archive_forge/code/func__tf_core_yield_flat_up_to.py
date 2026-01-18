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
def _tf_core_yield_flat_up_to(shallow_tree, input_tree, is_nested_fn, path=()):
    """Yields (path, value) pairs of input_tree flattened up to shallow_tree.

  Args:
    shallow_tree: Nested structure. Traverse no further than its leaf nodes.
    input_tree: Nested structure. Return the paths and values from this tree.
      Must have the same upper structure as shallow_tree.
    is_nested_fn: Function used to test if a value should be treated as a nested
      structure.
    path: Tuple. Optional argument, only used when recursing. The path from the
      root of the original shallow_tree, down to the root of the shallow_tree
      arg of this recursive call.

  Yields:
    Pairs of (path, value), where path the tuple path of a leaf node in
    shallow_tree, and value is the value of the corresponding node in
    input_tree.
  """
    if not is_nested_fn(shallow_tree):
        yield (path, input_tree)
    else:
        input_tree = dict(_tf_core_yield_sorted_items(input_tree))
        for shallow_key, shallow_subtree in _tf_core_yield_sorted_items(shallow_tree):
            subpath = path + (shallow_key,)
            input_subtree = input_tree[shallow_key]
            for leaf_path, leaf_value in _tf_core_yield_flat_up_to(shallow_subtree, input_subtree, is_nested_fn, path=subpath):
                yield (leaf_path, leaf_value)