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
def _tf_data_map_structure_up_to(shallow_tree, func, *inputs):
    if not inputs:
        raise ValueError('Argument `inputs` is empty. Cannot map over no sequences.')
    for input_tree in inputs:
        _tf_data_assert_shallow_structure(shallow_tree, input_tree)
    all_flattened_up_to = (_tf_data_flatten_up_to(shallow_tree, input_tree) for input_tree in inputs)
    results = [func(*tensors) for tensors in zip(*all_flattened_up_to)]
    return _tf_data_pack_sequence_as(structure=shallow_tree, flat_sequence=results)