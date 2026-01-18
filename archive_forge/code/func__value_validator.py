import copy
import sys
from weakref import ref
from traits.observation.i_observable import IObservable
from traits.trait_base import Undefined, _validate_everything
from traits.trait_errors import TraitError
def _value_validator(self, value):
    """ Calls the trait's value_handler.validate

        Parameters
        ----------
        value : any
            The value to validate.

        Returns
        -------
        validated_value : any
            The validated value.

        Raises
        ------
        TraitError
            If the validation fails.
        """
    trait = getattr(self, 'trait', None)
    object = getattr(self, 'object', lambda: None)()
    if trait is None or object is None:
        return value
    validate = trait.value_handler.validate
    if validate is None:
        return value
    try:
        return validate(object, self.name, value)
    except TraitError as excep:
        excep.set_prefix('Each value of the')
        raise excep