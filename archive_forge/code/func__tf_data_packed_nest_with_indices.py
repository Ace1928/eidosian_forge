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
def _tf_data_packed_nest_with_indices(structure, flat, index):
    """Helper function for pack_nest_as.

  Args:
    structure: Substructure (tuple of elements and/or tuples) to mimic
    flat: Flattened values to output substructure for.
    index: Index at which to start reading from flat.

  Returns:
    The tuple (new_index, child), where:
      * new_index - the updated index into `flat` having processed `structure`.
      * packed - the subset of `flat` corresponding to `structure`,
                 having started at `index`, and packed into the same nested
                 format.

  Raises:
    ValueError: if `structure` contains more elements than `flat`
      (assuming indexing starts from `index`).
  """
    packed = []
    for s in _tf_data_yield_value(structure):
        if _tf_data_is_nested(s):
            new_index, child = _tf_data_packed_nest_with_indices(s, flat, index)
            packed.append(sequence_like(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return (index, packed)