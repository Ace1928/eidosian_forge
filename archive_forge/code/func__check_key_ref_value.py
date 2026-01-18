import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _check_key_ref_value(self, key, references, value):
    """Check that 'key' and 'references' are all valid.

        :param key: A key tuple. Must conform to the key interface (be a tuple,
            be of the right length, not have any whitespace or nulls in any key
            element.)
        :param references: An iterable of reference lists. Something like
            [[(ref, key)], [(ref, key), (other, key)]]
        :param value: The value associate with this key. Must not contain
            newlines or null characters.
        :return: (node_refs, absent_references)

            * node_refs: basically a packed form of 'references' where all
              iterables are tuples
            * absent_references: reference keys that are not in self._nodes.
              This may contain duplicates if the same key is referenced in
              multiple lists.
        """
    as_st = StaticTuple.from_sequence
    self._check_key(key)
    if _newline_null_re.search(value) is not None:
        raise BadIndexValue(value)
    if len(references) != self.reference_lists:
        raise BadIndexValue(references)
    node_refs = []
    absent_references = []
    for reference_list in references:
        for reference in reference_list:
            if reference not in self._nodes:
                self._check_key(reference)
                absent_references.append(reference)
        reference_list = as_st([as_st(ref).intern() for ref in reference_list])
        node_refs.append(reference_list)
    return (as_st(node_refs), absent_references)