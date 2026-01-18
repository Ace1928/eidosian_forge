import copy
import copyreg
from itertools import chain
from weakref import ref
from traits.observation.i_observable import IObservable
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
@IObservable.register
class TraitSet(set):
    """ A subclass of set that validates and notifies listeners of changes.

    Parameters
    ----------
    value : iterable, optional
        Iterable providing the items for the set.
    item_validator : callable, optional
        Called to validate and/or transform items added to the set. The
        callable should accept a single item and return the transformed
        item, raising TraitError for invalid items. If not given, no
        item validation is performed.
    notifiers : list of callable, optional
        A list of callables with the signature::

            notifier(trait_set, removed, added)

        Where 'added' is a set containing new values that have been added.
        And 'removed' is a set containing old values that have been removed.

        If this argument is not given, the list of notifiers is initially
        empty.

    Attributes
    ----------
    item_validator : callable
        Called to validate and/or transform items added to the set. The
        callable should accept a single item and return the transformed
        item, raising TraitError for invalid items.
    notifiers : list of callable
        A list of callables with the signature::

            notifier(trait_set, removed, added)

        where 'added' is a set containing new values that have been added
        and 'removed' is a set containing old values that have been removed.
    """

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        self.item_validator = _validate_everything
        self.notifiers = []
        return self

    def __init__(self, value=(), *, item_validator=None, notifiers=None):
        if item_validator is not None:
            self.item_validator = item_validator
        super().__init__((self.item_validator(item) for item in value))
        if notifiers is not None:
            self.notifiers = notifiers

    def notify(self, removed, added):
        """ Call all notifiers.

        This simply calls all notifiers provided by the class, if any.
        The notifiers are expected to have the signature::

            notifier(trait_set, removed, added)

        Any return values are ignored. Any exceptions raised are not
        handled. Notifiers are therefore expected not to raise any
        exceptions under normal use.

        Parameters
        ----------
        removed : set
            The items that have been removed.
        added : set
            The new items that have been added to the set.
        """
        for notifier in self.notifiers:
            notifier(self, removed, added)

    def __iand__(self, value):
        """  Return self &= value.

        Parameters
        ----------
        value : set or frozenset
            A value.

        Returns
        -------
        self : TraitSet
            The updated set.
        """
        old_set = self.copy()
        retval = super().__iand__(value)
        removed = old_set.difference(self)
        if len(removed) > 0:
            self.notify(removed, set())
        return retval

    def __ior__(self, value):
        """ Return self |= value.

        Parameters
        ----------
        value : set or frozenset
            A value.

        Returns
        -------
        self : TraitSet
            The updated set.
        """
        old_set = self.copy()
        if isinstance(value, (set, frozenset)):
            value = {self.item_validator(item) for item in value}
        retval = super().__ior__(value)
        added = self.difference(old_set)
        if len(added) > 0:
            self.notify(set(), added)
        return retval

    def __isub__(self, value):
        """ Return self-=value.

        Parameters
        ----------
        value : set or frozenset
            A value.

        Returns
        -------
        self : TraitSet
            The updated set.
        """
        old_set = self.copy()
        retval = super().__isub__(value)
        removed = old_set.difference(self)
        if len(removed) > 0:
            self.notify(removed, set())
        return retval

    def __ixor__(self, value):
        """ Return self ^= value.

        Parameters
        ----------
        value : set or frozenset
            A value.

        Returns
        -------
        self : TraitSet
            The updated set.
        """
        removed = set()
        added = set()
        if isinstance(value, (set, frozenset)):
            values = set(value)
            removed = self.intersection(values)
            raw_added = values.difference(removed)
            validated_added = {self.item_validator(item) for item in raw_added}
            added = validated_added.difference(self)
            value = added | removed
        retval = super().__ixor__(value)
        if removed or added:
            self.notify(removed, added)
        return retval

    def add(self, value):
        """ Add an element to a set.

        This has no effect if the element is already present.

        Parameters
        ----------
        value : any
            The value to add to the set.
        """
        value = self.item_validator(value)
        value_in_self = value in self
        super().add(value)
        if not value_in_self:
            self.notify(set(), {value})

    def clear(self):
        """ Remove all elements from this set. """
        removed = set(self)
        super().clear()
        if removed:
            self.notify(removed, set())

    def discard(self, value):
        """ Remove an element from the set if it is a member.

        If the element is not a member, do nothing.

        Parameters
        ----------
        value : any
            An item in the set
        """
        value_in_self = value in self
        super().discard(value)
        if value_in_self:
            self.notify({value}, set())

    def difference_update(self, *args):
        """  Remove all elements of another set from this set.

        Parameters
        ----------
        args : iterables
            The other iterables.
        """
        old_set = self.copy()
        super().difference_update(*args)
        removed = old_set.difference(self)
        if len(removed) > 0:
            self.notify(removed, set())

    def intersection_update(self, *args):
        """  Update the set with the intersection of itself and another set.

        Parameters
        ----------
        args : iterables
            The other iterables.
        """
        old_set = self.copy()
        super().intersection_update(*args)
        removed = old_set.difference(self)
        if len(removed) > 0:
            self.notify(removed, set())

    def pop(self):
        """ Remove and return an arbitrary set element.

        Raises KeyError if the set is empty.

        Returns
        -------
        item : any
            An element from the set.

        Raises
        ------
        KeyError
            If the set is empty.
        """
        removed = super().pop()
        self.notify({removed}, set())
        return removed

    def remove(self, value):
        """ Remove an element that is a member of the set.

        If the element is not a member, raise a KeyError.

        Parameters
        ----------
        value : any
            An element in the set

        Raises
        ------
        KeyError
            If the value is not found in the set.
        """
        super().remove(value)
        self.notify({value}, set())

    def symmetric_difference_update(self, value):
        """ Update the set with the symmetric difference of itself and another.

        Parameters
        ----------
        value : iterable
        """
        values = set(value)
        removed = self.intersection(values)
        raw_result = values.difference(removed)
        validated_result = {self.item_validator(item) for item in raw_result}
        added = validated_result.difference(self)
        super().symmetric_difference_update(removed | added)
        if removed or added:
            self.notify(removed, added)

    def update(self, *args):
        """ Update the set with the union of itself and others.

        Parameters
        ----------
        args : iterables
            The other iterables.
        """
        validated_values = {self.item_validator(item) for item in chain.from_iterable(args)}
        added = validated_values.difference(self)
        super().update(added)
        if len(added) > 0:
            self.notify(set(), added)

    def __deepcopy__(self, memo):
        """ Perform a deepcopy operation.

        Notifiers are transient and should not be copied.
        """
        result = TraitSet([copy.deepcopy(x, memo) for x in self], item_validator=copy.deepcopy(self.validator, memo), notifiers=[])
        return result

    def __getstate__(self):
        """ Get the state of the object for serialization.

        Notifiers are transient and should not be serialized.
        """
        result = self.__dict__.copy()
        del result['notifiers']
        return result

    def __setstate__(self, state):
        """ Restore the state of the object after serialization.

        Notifiers are transient and are restored to the empty list.
        """
        state['notifiers'] = []
        self.__dict__.update(state)

    def _notifiers(self, force_create):
        """ Return a list of callables where each callable is a notifier.
        The list is expected to be mutated for contributing or removing
        notifiers from the object.

        Parameters
        ----------
        force_create: boolean
            Not used here.
        """
        return self.notifiers