import copy
import sys
from weakref import ref
from traits.observation.i_observable import IObservable
from traits.trait_base import Undefined, _validate_everything
from traits.trait_errors import TraitError
class TraitDictEvent(object):
    """ An object reporting in-place changes to a traits dict.

    Parameters
    ----------
    removed : dict, optional
        Old keys and values that were just removed.
    added : dict, optional
        New keys and values that were just added.
    changed : dict, optional
        Updated keys and their previous values.

    Attributes
    ----------
    removed : dict
        Old keys and values that were just removed.
    added : dict
        New keys and values that were just added.
    changed : dict
        Updated keys and their previous values.
    """

    def __init__(self, *, removed=None, added=None, changed=None):
        if removed is None:
            removed = {}
        self.removed = removed
        if added is None:
            added = {}
        self.added = added
        if changed is None:
            changed = {}
        self.changed = changed

    def __repr__(self):
        return f'{self.__class__.__name__}(removed={self.removed!r}, added={self.added!r}, changed={self.changed!r})'