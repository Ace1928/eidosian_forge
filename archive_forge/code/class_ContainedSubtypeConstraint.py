import sys
from pyasn1.type import error
class ContainedSubtypeConstraint(AbstractConstraint):
    """Create a ContainedSubtypeConstraint object.

    The ContainedSubtypeConstraint satisfies any value that
    is present in the set of permitted values and also
    satisfies included constraints.

    The ContainedSubtypeConstraint object can be applied to
    any ASN.1 type.

    Parameters
    ----------
    \\*values:
        Full set of values and constraint objects permitted
        by this constraint object.

    Examples
    --------
    .. code-block:: python

        class DivisorOfEighteen(Integer):
            '''
            ASN.1 specification:

            Divisors-of-18 ::= INTEGER (INCLUDES Divisors-of-6 | 9 | 18)
            '''
            subtypeSpec = ContainedSubtypeConstraint(
                SingleValueConstraint(1, 2, 3, 6), 9, 18
            )

        # this will succeed
        divisor_of_eighteen = DivisorOfEighteen(9)

        # this will raise ValueConstraintError
        divisor_of_eighteen = DivisorOfEighteen(10)
    """

    def _testValue(self, value, idx):
        for constraint in self._values:
            if isinstance(constraint, AbstractConstraint):
                constraint(value, idx)
            elif value not in self._set:
                raise error.ValueConstraintError(value)