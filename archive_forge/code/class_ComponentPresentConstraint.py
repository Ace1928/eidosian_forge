import sys
from pyasn1.type import error
class ComponentPresentConstraint(AbstractConstraint):
    """Create a ComponentPresentConstraint object.

    The ComponentPresentConstraint is only satisfied when the value
    is not `None`.

    The ComponentPresentConstraint object is typically used with
    `WithComponentsConstraint`.

    Examples
    --------
    .. code-block:: python

        present = ComponentPresentConstraint()

        # this will succeed
        present('whatever')

        # this will raise ValueConstraintError
        present(None)
    """

    def _setValues(self, values):
        self._values = ('<must be present>',)
        if values:
            raise error.PyAsn1Error('No arguments expected')

    def _testValue(self, value, idx):
        if value is None:
            raise error.ValueConstraintError('Component is not present:')