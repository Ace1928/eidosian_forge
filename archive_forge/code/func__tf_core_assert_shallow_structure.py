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
def _tf_core_assert_shallow_structure(shallow_tree, input_tree, check_types=True, expand_composites=False):
    is_nested_fn = _is_nested_or_composite if expand_composites else _tf_core_is_nested
    if is_nested_fn(shallow_tree):
        if not is_nested_fn(input_tree):
            raise TypeError('If shallow structure is a sequence, input must also be a sequence. Input has type: %s.' % type(input_tree))
        if isinstance(shallow_tree, _wrapt.ObjectProxy):
            shallow_type = type(shallow_tree.__wrapped__)
        else:
            shallow_type = type(shallow_tree)
        if check_types and (not isinstance(input_tree, shallow_type)):
            shallow_is_namedtuple = is_namedtuple(shallow_tree, False)
            input_is_namedtuple = is_namedtuple(input_tree, False)
            if shallow_is_namedtuple and input_is_namedtuple:
                if not same_namedtuples(shallow_tree, input_tree):
                    raise TypeError(STRUCTURES_HAVE_MISMATCHING_TYPES.format(input_type=type(input_tree), shallow_type=type(shallow_tree)))
            elif isinstance(shallow_tree, list) and isinstance(input_tree, list):
                pass
            elif (_is_composite_tensor(shallow_tree) or _is_type_spec(shallow_tree)) and (_is_composite_tensor(input_tree) or _is_type_spec(input_tree)):
                pass
            elif not (isinstance(shallow_tree, _collections_abc.Mapping) and isinstance(input_tree, _collections_abc.Mapping)):
                raise TypeError(STRUCTURES_HAVE_MISMATCHING_TYPES.format(input_type=type(input_tree), shallow_type=type(shallow_tree)))
        if _is_composite_tensor(shallow_tree) or _is_composite_tensor(input_tree):
            if not ((_is_composite_tensor(input_tree) or _is_type_spec(input_tree)) and (_is_composite_tensor(shallow_tree) or _is_type_spec(shallow_tree))):
                raise TypeError(STRUCTURES_HAVE_MISMATCHING_TYPES.format(input_type=type(input_tree), shallow_type=type(shallow_tree)))
            type_spec_1 = (shallow_tree if _is_type_spec(shallow_tree) else shallow_tree._type_spec)._without_tensor_names()
            type_spec_2 = (input_tree if _is_type_spec(input_tree) else input_tree._type_spec)._without_tensor_names()
            if hasattr(type_spec_1, '_get_structure') and hasattr(type_spec_2, '_get_structure'):
                result = type_spec_1._get_structure() == type_spec_2._get_structure() or None
            else:
                result = type_spec_1.most_specific_common_supertype([type_spec_2])
            if result is None:
                raise ValueError('Incompatible CompositeTensor TypeSpecs: %s vs. %s' % (type_spec_1, type_spec_2))
        elif _is_type_spec(shallow_tree):
            if not _is_type_spec(input_tree):
                raise TypeError('If shallow structure is a TypeSpec, input must also be a TypeSpec.  Input has type: %s.' % type(input_tree))
        elif len(input_tree) != len(shallow_tree):
            raise ValueError(STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(input_length=len(input_tree), shallow_length=len(shallow_tree)))
        elif len(input_tree) < len(shallow_tree):
            raise ValueError(INPUT_TREE_SMALLER_THAN_SHALLOW_TREE.format(input_size=len(input_tree), shallow_size=len(shallow_tree)))
        if isinstance(shallow_tree, _collections_abc.Mapping):
            absent_keys = set(shallow_tree) - set(input_tree)
            if absent_keys:
                raise ValueError(SHALLOW_TREE_HAS_INVALID_KEYS.format(sorted(absent_keys)))
        for shallow_branch, input_branch in zip(_tf_core_yield_value(shallow_tree), _tf_core_yield_value(input_tree)):
            _tf_core_assert_shallow_structure(shallow_branch, input_branch, check_types=check_types, expand_composites=expand_composites)