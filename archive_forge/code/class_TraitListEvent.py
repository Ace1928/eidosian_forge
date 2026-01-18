import copy
import operator
from weakref import ref
from traits.observation.i_observable import IObservable
from traits.trait_base import class_of, Undefined, _validate_everything
from traits.trait_errors import TraitError
class TraitListEvent(object):
    """ An object reporting in-place changes to a trait list.

    Parameters
    ----------
    index : int or slice, optional
        An index or slice indicating the location of the changes to the trait
        list. The default is 0.
    added : list, optional
        The list of values added to the trait list.
    removed : list, optional
        The list of values removed from the list.

    Attributes
    ----------
    index : int or slice
        An index or slice indicating the location of the changes to the list.
    added : list
        The list of values added to the list.  If nothing was added this is
        an empty list.
    removed : list
        The list of values removed from the list.  If nothing was removed
        this is an empty list.
    """

    def __init__(self, *, index=0, removed=None, added=None):
        self.index = index
        if removed is None:
            removed = []
        self.removed = removed
        if added is None:
            added = []
        self.added = added

    def __repr__(self):
        return f'{self.__class__.__name__}(index={self.index!r}, removed={self.removed!r}, added={self.added!r})'