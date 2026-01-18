import copy
import copyreg
from itertools import chain
from weakref import ref
from traits.observation.i_observable import IObservable
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
class TraitSetObject(TraitSet):
    """ A specialization of TraitSet with a default validator and notifier
    for compatibility with Traits versions before 6.0.

    Parameters
    ----------
    trait : CTrait
        The trait that the set has been assigned to.
    object : HasTraits
        The object the set belongs to.
    name : str
        The name of the trait on the object.
    value : iterable
        The initial value of the set.

    Attributes
    ----------
    trait : CTrait
        The trait that the set has been assigned to.
    object : HasTraits
        The object the set belongs to.
    name : str
        The name of the trait on the object.
    value : iterable
        The initial value of the set.
    """

    def __init__(self, trait, object, name, value):
        self.trait = trait
        self.object = ref(object)
        self.name = name
        self.name_items = None
        if trait.has_items:
            self.name_items = name + '_items'
        super().__init__(value, item_validator=self._validator, notifiers=[self.notifier])

    def _validator(self, value):
        """ Validates the value by calling the inner trait's validate method.

        Parameters
        ----------
        value : any
            The value to be validated.

        Returns
        -------
        value : any
            The validated value.

        Raises
        ------
        TraitError
            On validation failure for the inner trait.
        """
        object_ref = getattr(self, 'object', None)
        trait = getattr(self, 'trait', None)
        if object_ref is None or trait is None:
            return value
        object = object_ref()
        validate = trait.item_trait.handler.validate
        if validate is None:
            return value
        try:
            return validate(object, self.name, value)
        except TraitError as excp:
            excp.set_prefix('Each element of the')
            raise excp

    def notifier(self, trait_set, removed, added):
        """ Converts and consolidates the parameters to a TraitSetEvent and
        then fires the event.

        Parameters
        ----------
        trait_set : set
            The complete set
        removed : set
            Set of values that were removed.
        added : set
            Set of values that were added.
        """
        if self.name_items is None:
            return
        object = self.object()
        if object is None:
            return
        if getattr(object, self.name) is not self:
            return
        event = TraitSetEvent(removed=removed, added=added)
        items_event = self.trait.items_event()
        object.trait_items_event(self.name_items, event, items_event)

    def __deepcopy__(self, memo):
        """ Perform a deepcopy operation.

        Notifiers are transient and should not be copied.
        """
        result = TraitSetObject(self.trait, lambda: None, self.name, {copy.deepcopy(x, memo) for x in self})
        return result

    def __getstate__(self):
        """ Get the state of the object for serialization.

        Notifiers are transient and should not be serialized.
        """
        result = super().__getstate__()
        del result['object']
        del result['trait']
        return result

    def __setstate__(self, state):
        """ Restore the state of the object after serialization.

        Notifiers are transient and are restored to the empty list.
        """
        state.setdefault('name', '')
        state['notifiers'] = [self.notifier]
        state['object'] = lambda: None
        state['trait'] = None
        self.__dict__.update(state)

    def __reduce_ex__(self, protocol=None):
        """ Overridden to make sure we call our custom __getstate__.
        """
        return (copyreg._reconstructor, (type(self), set, list(self)), self.__getstate__())