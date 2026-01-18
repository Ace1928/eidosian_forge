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
def _tf_core_map_structure(func, *structure, **kwargs):
    if not callable(func):
        raise TypeError('func must be callable, got: %s' % func)
    if not structure:
        raise ValueError('Must provide at least one structure')
    check_types = kwargs.pop('check_types', True)
    expand_composites = kwargs.pop('expand_composites', False)
    if kwargs:
        raise ValueError('Only valid keyword arguments are `check_types` and `expand_composites`, not: `%s`' % '`, `'.join(kwargs.keys()))
    for other in structure[1:]:
        _tf_core_assert_same_structure(structure[0], other, check_types=check_types, expand_composites=expand_composites)
    flat_structure = (_tf_core_flatten(s, expand_composites) for s in structure)
    entries = zip(*flat_structure)
    return _tf_core_pack_sequence_as(structure[0], [func(*x) for x in entries], expand_composites=expand_composites)