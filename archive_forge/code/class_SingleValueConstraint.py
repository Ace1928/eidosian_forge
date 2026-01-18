import sys
from pyasn1.type import error
class SingleValueConstraint(AbstractConstraint):
    """Create a SingleValueConstraint object.

    The SingleValueConstraint satisfies any value that
    is present in the set of permitted values.

    The SingleValueConstraint object can be applied to
    any ASN.1 type.

    Parameters
    ----------
    \\*values: :class:`int`
        Full set of values permitted by this constraint object.

    Examples
    --------
    .. code-block:: python

        class DivisorOfSix(Integer):
            '''
            ASN.1 specification:

            Divisor-Of-6 ::= INTEGER (1 | 2 | 3 | 6)
            '''
            subtypeSpec = SingleValueConstraint(1, 2, 3, 6)

        # this will succeed
        divisor_of_six = DivisorOfSix(1)

        # this will raise ValueConstraintError
        divisor_of_six = DivisorOfSix(7)
    """

    def _setValues(self, values):
        self._values = values
        self._set = set(values)

    def _testValue(self, value, idx):
        if value not in self._set:
            raise error.ValueConstraintError(value)