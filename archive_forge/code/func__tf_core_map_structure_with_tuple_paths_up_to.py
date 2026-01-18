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
def _tf_core_map_structure_with_tuple_paths_up_to(shallow_tree, func, *inputs, **kwargs):
    """See comments for map_structure_with_tuple_paths_up_to() in tensorflow/python/util/nest.py."""
    if not inputs:
        raise ValueError('Cannot map over no sequences')
    check_types = kwargs.pop('check_types', True)
    expand_composites = kwargs.pop('expand_composites', False)
    is_nested_fn = _is_nested_or_composite if expand_composites else _tf_core_is_nested
    for input_tree in inputs:
        _tf_core_assert_shallow_structure(shallow_tree, input_tree, check_types=check_types, expand_composites=expand_composites)
    flat_value_gen = (_tf_core_flatten_up_to(shallow_tree, input_tree, check_types, expand_composites=expand_composites) for input_tree in inputs)
    flat_path_gen = (path for path, _ in _tf_core_yield_flat_up_to(shallow_tree, inputs[0], is_nested_fn))
    results = [func(*args, **kwargs) for args in zip(flat_path_gen, *flat_value_gen)]
    return _tf_core_pack_sequence_as(structure=shallow_tree, flat_sequence=results, expand_composites=expand_composites)