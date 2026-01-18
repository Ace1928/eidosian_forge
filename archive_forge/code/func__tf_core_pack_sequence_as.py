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
def _tf_core_pack_sequence_as(structure, flat_sequence, expand_composites, sequence_fn=None):
    """Implements sequence packing, with the option to alter the structure."""
    is_nested_fn = _is_nested_or_composite if expand_composites else _tf_core_is_nested
    sequence_fn = sequence_fn or sequence_like

    def truncate(value, length):
        value_str = str(value)
        return value_str[:length] + (value_str[length:] and '...')
    if not is_nested_fn(flat_sequence):
        raise TypeError('Attempted to pack value:\n  {}\ninto a structure, but found incompatible type `{}` instead.'.format(truncate(flat_sequence, 100), type(flat_sequence)))
    if not is_nested_fn(structure):
        if len(flat_sequence) != 1:
            raise ValueError('The target structure is of type `{}`\n  {}\nHowever the input is a sequence ({}) of length {}.\n  {}\nnest cannot guarantee that it is safe to map one to the other.'.format(type(structure), truncate(structure, 100), type(flat_sequence), len(flat_sequence), truncate(flat_sequence, 100)))
        return flat_sequence[0]
    try:
        final_index, packed = _tf_core_packed_nest_with_indices(structure, flat_sequence, 0, is_nested_fn, sequence_fn)
        if final_index < len(flat_sequence):
            raise IndexError
    except IndexError:
        flat_structure = _tf_core_flatten(structure, expand_composites=expand_composites)
        if len(flat_structure) != len(flat_sequence):
            raise ValueError('Could not pack sequence. Structure had %d atoms, but flat_sequence had %d items.  Structure: %s, flat_sequence: %s.' % (len(flat_structure), len(flat_sequence), structure, flat_sequence))
    return sequence_fn(structure, packed)