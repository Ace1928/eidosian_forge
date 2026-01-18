import copy
import sys
from weakref import ref
from traits.observation.i_observable import IObservable
from traits.trait_base import Undefined, _validate_everything
from traits.trait_errors import TraitError
def _key_validator(self, key):
    """ Calls the trait's key_trait.handler.validate.

        Parameters
        ----------
        key : A hashable object.
            The key to validate.

        Returns
        -------
        validated_key : A hashable object.
            The validated key.

        Raises
        ------
        TraitError
            If the validation fails.
        """
    trait = getattr(self, 'trait', None)
    object = getattr(self, 'object', lambda: None)()
    if trait is None or object is None:
        return key
    validate = trait.key_trait.handler.validate
    if validate is None:
        return key
    try:
        return validate(object, self.name, key)
    except TraitError as excep:
        excep.set_prefix('Each key of the')
        raise excep