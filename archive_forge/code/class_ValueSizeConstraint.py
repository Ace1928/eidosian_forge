import sys
from pyasn1.type import error
class ValueSizeConstraint(ValueRangeConstraint):
    """Create a ValueSizeConstraint object.

    The ValueSizeConstraint satisfies any value for
    as long as its size falls within the range of
    permitted sizes.

    The ValueSizeConstraint object can be applied
    to :class:`~pyasn1.type.univ.BitString`,
    :class:`~pyasn1.type.univ.OctetString` (including
    all :ref:`character ASN.1 types <type.char>`),
    :class:`~pyasn1.type.univ.SequenceOf`
    and :class:`~pyasn1.type.univ.SetOf` types.

    Parameters
    ----------
    minimum: :class:`int`
        Minimum permitted size of the value (inclusive)

    maximum: :class:`int`
        Maximum permitted size of the value (inclusive)

    Examples
    --------
    .. code-block:: python

        class BaseballTeamRoster(SetOf):
            '''
            ASN.1 specification:

            BaseballTeamRoster ::= SET SIZE (1..25) OF PlayerNames
            '''
            componentType = PlayerNames()
            subtypeSpec = ValueSizeConstraint(1, 25)

        # this will succeed
        team = BaseballTeamRoster()
        team.extend(['Jan', 'Matej'])
        encode(team)

        # this will raise ValueConstraintError
        team = BaseballTeamRoster()
        team.extend(['Jan'] * 26)
        encode(team)

    Note
    ----
    Whenever ValueSizeConstraint is applied to mutable types
    (e.g. :class:`~pyasn1.type.univ.SequenceOf`,
    :class:`~pyasn1.type.univ.SetOf`), constraint
    validation only happens at the serialisation phase rather
    than schema instantiation phase (as it is with immutable
    types).
    """

    def _testValue(self, value, idx):
        valueSize = len(value)
        if valueSize < self.start or valueSize > self.stop:
            raise error.ValueConstraintError(value)