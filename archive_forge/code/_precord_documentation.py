from pyrsistent._checked_types import CheckedType, _restore_pickle, InvariantException, store_invariants
from pyrsistent._field_common import (
from pyrsistent._pmap import PMap, pmap

        Serialize the current PRecord using custom serializer functions for fields where
        such have been supplied.
        