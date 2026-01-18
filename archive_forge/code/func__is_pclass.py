from pyrsistent._checked_types import (InvariantException, CheckedType, _restore_pickle, store_invariants)
from pyrsistent._field_common import (
from pyrsistent._transformations import transform
def _is_pclass(bases):
    return len(bases) == 1 and bases[0] == CheckedType