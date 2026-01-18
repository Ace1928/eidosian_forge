import copy
import operator
from weakref import ref
from traits.observation.i_observable import IObservable
from traits.trait_base import class_of, Undefined, _validate_everything
from traits.trait_errors import TraitError
def _removed_items(items, index, return_for_invalid_index):
    """
    Return removed items for a given list and index, suppressing IndexError.

    This is used by the __setitem__ and __delitem__ implementations to
    get the "removed" part of the event.

    Note that this deliberately suppresses any IndexError arising from
    an out-of-range integer index. A suitable IndexError will be re-raised
    when the actual __delitem__ or __setitem__ operation is performed.

    Parameters
    ----------
    items : list
        The list being operated on.
    index : integer or slice
        Index of items to remove or replace.
    return_for_invalid_index : any
        Object to return for an invalid index.

    Returns
    -------
    removed_items : list
        List containing the removed items.
    """
    if isinstance(index, slice):
        return items[index]
    else:
        try:
            return [items[index]]
        except IndexError:
            return return_for_invalid_index