from collections import abc as collections_abc
import logging
import sys
from typing import Mapping, Sequence, Text, TypeVar, Union
from .sequence import _is_attrs
from .sequence import _is_namedtuple
from .sequence import _sequence_like
from .sequence import _sorted
def _packed_nest_with_indices(structure, flat, index):
    """Helper function for ``unflatten_as``.

  Args:
    structure: Substructure (list / tuple / dict) to mimic.
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
    for s in _yield_value(structure):
        if is_nested(s):
            new_index, child = _packed_nest_with_indices(s, flat, index)
            packed.append(_sequence_like(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return (index, packed)