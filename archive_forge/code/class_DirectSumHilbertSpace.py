from functools import reduce
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.sets.sets import Interval
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.qexpr import QuantumError
class DirectSumHilbertSpace(HilbertSpace):
    """A direct sum of Hilbert spaces [1]_.

    This class uses the ``+`` operator to represent direct sums between
    different Hilbert spaces.

    A ``DirectSumHilbertSpace`` object takes in an arbitrary number of
    ``HilbertSpace`` objects as its arguments. Also, addition of
    ``HilbertSpace`` objects will automatically return a direct sum object.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace

    >>> c = ComplexSpace(2)
    >>> f = FockSpace()
    >>> hs = c+f
    >>> hs
    C(2)+F
    >>> hs.dimension
    oo
    >>> list(hs.spaces)
    [C(2), F]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space#Direct_sums
    """

    def __new__(cls, *args):
        r = cls.eval(args)
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, *args)
        return obj

    @classmethod
    def eval(cls, args):
        """Evaluates the direct product."""
        new_args = []
        recall = False
        for arg in args:
            if isinstance(arg, DirectSumHilbertSpace):
                new_args.extend(arg.args)
                recall = True
            elif isinstance(arg, HilbertSpace):
                new_args.append(arg)
            else:
                raise TypeError('Hilbert spaces can only be summed with other                 Hilbert spaces: %r' % arg)
        if recall:
            return DirectSumHilbertSpace(*new_args)
        else:
            return None

    @property
    def dimension(self):
        arg_list = [arg.dimension for arg in self.args]
        if S.Infinity in arg_list:
            return S.Infinity
        else:
            return reduce(lambda x, y: x + y, arg_list)

    @property
    def spaces(self):
        """A tuple of the Hilbert spaces in this direct sum."""
        return self.args

    def _sympyrepr(self, printer, *args):
        spaces_reprs = [printer._print(arg, *args) for arg in self.args]
        return 'DirectSumHilbertSpace(%s)' % ','.join(spaces_reprs)

    def _sympystr(self, printer, *args):
        spaces_strs = [printer._print(arg, *args) for arg in self.args]
        return '+'.join(spaces_strs)

    def _pretty(self, printer, *args):
        length = len(self.args)
        pform = printer._print('', *args)
        for i in range(length):
            next_pform = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (DirectSumHilbertSpace, TensorProductHilbertSpace)):
                next_pform = prettyForm(*next_pform.parens(left='(', right=')'))
            pform = prettyForm(*pform.right(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right(' ⊕ '))
                else:
                    pform = prettyForm(*pform.right(' + '))
        return pform

    def _latex(self, printer, *args):
        length = len(self.args)
        s = ''
        for i in range(length):
            arg_s = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (DirectSumHilbertSpace, TensorProductHilbertSpace)):
                arg_s = '\\left(%s\\right)' % arg_s
            s = s + arg_s
            if i != length - 1:
                s = s + '\\oplus '
        return s