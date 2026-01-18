from sympy.core import Basic, Integer
import operator
class OrdinalOmega(Ordinal):
    """The ordinal omega which forms the base of all ordinals in cantor normal form.

    OrdinalOmega can be imported as ``omega``.

    Examples
    ========

    >>> from sympy.sets.ordinals import omega
    >>> omega + omega
    w*2
    """

    def __new__(cls):
        return Ordinal.__new__(cls)

    @property
    def terms(self):
        return (OmegaPower(1, 1),)