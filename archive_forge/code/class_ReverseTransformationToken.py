from pyomo.common import Factory
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import MouseTrap
from pyomo.common.deprecation import deprecated
from pyomo.common.modeling import unique_component_name
from pyomo.common.timing import TransformationTimer
class ReverseTransformationToken(object):
    """
    Class returned by reversible transformations' apply_to methods that
    can be passed back to the transformation in order to revert its changes
    to the model.

    We store the transformation that created it, so that we have some basic
    error checking when the user attempts to revert, and we store a dictionary
    that can be whatever the transformation wants/needs in order to revert
    itself.

    args:
        transformation: The class of the transformation that created this token
        model: The model being transformed when this token was created
        targets: The targets on 'model' being transformed when this token
                 was created.
        reverse_dict: Dictionary with everything the transformation needs to
                      undo itself.
    """

    def __init__(self, transformation, model, targets, reverse_dict):
        self._transformation = transformation
        self._model = model
        self._targets = ComponentSet(targets)
        self._reverse_dict = reverse_dict

    @property
    def transformation(self):
        return self._transformation

    @property
    def reverse_dict(self):
        return self._reverse_dict

    def check_token_valid(self, cls, model, targets):
        if cls is not self._transformation:
            raise ValueError("Attempting to reverse transformation of class '%s' using a token created by a transformation of class '%s'. Cannot revert transformation with a token from another transformation." % (cls, self._transformation))
        if model is not self._model:
            raise MouseTrap("A reverse transformation was called on model '%s', but the transformation that created this token was created from model '%s'. We do not currently support reversing transformations on clones of the transformed model." % (model.name, self._model.name))