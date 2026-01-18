from functools import reduce
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.sets.sets import Interval
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.qexpr import QuantumError
class TensorPowerHilbertSpace(HilbertSpace):
    """An exponentiated Hilbert space [1]_.

    Tensor powers (repeated tensor products) are represented by the
    operator ``**`` Identical Hilbert spaces that are multiplied together
    will be automatically combined into a single tensor power object.

    Any Hilbert space, product, or sum may be raised to a tensor power. The
    ``TensorPowerHilbertSpace`` takes two arguments: the Hilbert space; and the
    tensor power (number).

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace
    >>> from sympy import symbols

    >>> n = symbols('n')
    >>> c = ComplexSpace(2)
    >>> hs = c**n
    >>> hs
    C(2)**n
    >>> hs.dimension
    2**n

    >>> c = ComplexSpace(2)
    >>> c*c
    C(2)**2
    >>> f = FockSpace()
    >>> c*f*f
    C(2)*F**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space#Tensor_products
    """

    def __new__(cls, *args):
        r = cls.eval(args)
        if isinstance(r, Basic):
            return r
        return Basic.__new__(cls, *r)

    @classmethod
    def eval(cls, args):
        new_args = (args[0], sympify(args[1]))
        exp = new_args[1]
        if exp is S.One:
            return args[0]
        if exp is S.Zero:
            return S.One
        if len(exp.atoms()) == 1:
            if not (exp.is_Integer and exp >= 0 or exp.is_Symbol):
                raise ValueError('Hilbert spaces can only be raised to                 positive integers or Symbols: %r' % exp)
        else:
            for power in exp.atoms():
                if not (power.is_Integer or power.is_Symbol):
                    raise ValueError('Tensor powers can only contain integers                     or Symbols: %r' % power)
        return new_args

    @property
    def base(self):
        return self.args[0]

    @property
    def exp(self):
        return self.args[1]

    @property
    def dimension(self):
        if self.base.dimension is S.Infinity:
            return S.Infinity
        else:
            return self.base.dimension ** self.exp

    def _sympyrepr(self, printer, *args):
        return 'TensorPowerHilbertSpace(%s,%s)' % (printer._print(self.base, *args), printer._print(self.exp, *args))

    def _sympystr(self, printer, *args):
        return '%s**%s' % (printer._print(self.base, *args), printer._print(self.exp, *args))

    def _pretty(self, printer, *args):
        pform_exp = printer._print(self.exp, *args)
        if printer._use_unicode:
            pform_exp = prettyForm(*pform_exp.left(prettyForm('⨂')))
        else:
            pform_exp = prettyForm(*pform_exp.left(prettyForm('x')))
        pform_base = printer._print(self.base, *args)
        return pform_base ** pform_exp

    def _latex(self, printer, *args):
        base = printer._print(self.base, *args)
        exp = printer._print(self.exp, *args)
        return '{%s}^{\\otimes %s}' % (base, exp)