import sys
from pyasn1.type import error
class PermittedAlphabetConstraint(SingleValueConstraint):
    """Create a PermittedAlphabetConstraint object.

    The PermittedAlphabetConstraint satisfies any character
    string for as long as all its characters are present in
    the set of permitted characters.

    The PermittedAlphabetConstraint object can only be applied
    to the :ref:`character ASN.1 types <type.char>` such as
    :class:`~pyasn1.type.char.IA5String`.

    Parameters
    ----------
    \\*alphabet: :class:`str`
        Full set of characters permitted by this constraint object.

    Examples
    --------
    .. code-block:: python

        class BooleanValue(IA5String):
            '''
            ASN.1 specification:

            BooleanValue ::= IA5String (FROM ('T' | 'F'))
            '''
            subtypeSpec = PermittedAlphabetConstraint('T', 'F')

        # this will succeed
        truth = BooleanValue('T')
        truth = BooleanValue('TF')

        # this will raise ValueConstraintError
        garbage = BooleanValue('TAF')
    """

    def _setValues(self, values):
        self._values = values
        self._set = set(values)

    def _testValue(self, value, idx):
        if not self._set.issuperset(value):
            raise error.ValueConstraintError(value)