import copy
import operator
from weakref import ref
from traits.observation.i_observable import IObservable
from traits.trait_base import class_of, Undefined, _validate_everything
from traits.trait_errors import TraitError
def _normalize_slice_or_index(index, length):
    """ Normalize a slice or index for use with __delitem__ or __setitem__.

    For slices with positive step, returns a slice that's equivalent for the
    purposes of __delitem__ and __setitem__ operations. For slices with
    negative step, a normalized slice representing the reverse of the given
    slice is returned: note that in this case, the matching *added* and
    *removed* lists will need to be reversed.

    Slices with a step of 1 or -1 are normalized to a single integer index,
    referring to the position of the first element referenced by the slice.

    Similarly, slices that refer to only a single element of the corresponding
    list (for example, a slice of `[1::10]` applied to a list of length 5)
    are normalized to the index that refers to that same element.

    Empty slices are also normalized to a single index. Note that in the case
    of an empty slice, the corresponding __delitem__ or __setitem__ operation
    does not cause any list change, so does not issue a notification. So the
    normalized index in this case is unused in current code.

    A normalized slice will have 0 <= start < stop <= length and a step >= 2.
    It should further satisfy start + step < stop. The stop will always be
    one larger than the last element referenced by the slice.

    For a plain integer index, it's assumed -length <= index < length on input
    (but this is not explicitly checked). A normalized output index will
    satisfy 0 <= index <= length.

    Parameters
    ----------
    index : slice or integer
        The slice to normalize
    length : int
        The length of the list to which the slice will be applied.

    Returns
    -------
    reversed : bool
        True if the returned slice is in the opposite direction to the
        original, else False.
    normalized_index : slice or integer
        An equivalent (or reversed equivalent) normalized slice or index.
    """
    if not isinstance(index, slice):
        index = operator.index(index)
        return (False, index + length if index < 0 else index)
    start, stop, step = index.indices(length)
    reversed = step < 0
    if reversed:
        start, stop, step = (min(stop - step + (start - stop) % step, length), start + 1, -step)
    stop -= (stop - start - 1) % step
    if step == 1 or stop - start <= step:
        return (reversed, start)
    else:
        return (reversed, slice(start, stop, step))