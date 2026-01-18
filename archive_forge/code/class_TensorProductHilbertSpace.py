from functools import reduce
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.sets.sets import Interval
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.qexpr import QuantumError
class TensorProductHilbertSpace(HilbertSpace):
    """A tensor product of Hilbert spaces [1]_.

    The tensor product between Hilbert spaces is represented by the
    operator ``*`` Products of the same Hilbert space will be combined into
    tensor powers.

    A ``TensorProductHilbertSpace`` object takes in an arbitrary number of
    ``HilbertSpace`` objects as its arguments. In addition, multiplication of
    ``HilbertSpace`` objects will automatically return this tensor product
    object.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace
    >>> from sympy import symbols

    >>> c = ComplexSpace(2)
    >>> f = FockSpace()
    >>> hs = c*f
    >>> hs
    C(2)*F
    >>> hs.dimension
    oo
    >>> hs.spaces
    (C(2), F)

    >>> c1 = ComplexSpace(2)
    >>> n = symbols('n')
    >>> c2 = ComplexSpace(n)
    >>> hs = c1*c2
    >>> hs
    C(2)*C(n)
    >>> hs.dimension
    2*n

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space#Tensor_products
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
            if isinstance(arg, TensorProductHilbertSpace):
                new_args.extend(arg.args)
                recall = True
            elif isinstance(arg, (HilbertSpace, TensorPowerHilbertSpace)):
                new_args.append(arg)
            else:
                raise TypeError('Hilbert spaces can only be multiplied by                 other Hilbert spaces: %r' % arg)
        comb_args = []
        prev_arg = None
        for new_arg in new_args:
            if prev_arg is not None:
                if isinstance(new_arg, TensorPowerHilbertSpace) and isinstance(prev_arg, TensorPowerHilbertSpace) and (new_arg.base == prev_arg.base):
                    prev_arg = new_arg.base ** (new_arg.exp + prev_arg.exp)
                elif isinstance(new_arg, TensorPowerHilbertSpace) and new_arg.base == prev_arg:
                    prev_arg = prev_arg ** (new_arg.exp + 1)
                elif isinstance(prev_arg, TensorPowerHilbertSpace) and new_arg == prev_arg.base:
                    prev_arg = new_arg ** (prev_arg.exp + 1)
                elif new_arg == prev_arg:
                    prev_arg = new_arg ** 2
                else:
                    comb_args.append(prev_arg)
                    prev_arg = new_arg
            elif prev_arg is None:
                prev_arg = new_arg
        comb_args.append(prev_arg)
        if recall:
            return TensorProductHilbertSpace(*comb_args)
        elif len(comb_args) == 1:
            return TensorPowerHilbertSpace(comb_args[0].base, comb_args[0].exp)
        else:
            return None

    @property
    def dimension(self):
        arg_list = [arg.dimension for arg in self.args]
        if S.Infinity in arg_list:
            return S.Infinity
        else:
            return reduce(lambda x, y: x * y, arg_list)

    @property
    def spaces(self):
        """A tuple of the Hilbert spaces in this tensor product."""
        return self.args

    def _spaces_printer(self, printer, *args):
        spaces_strs = []
        for arg in self.args:
            s = printer._print(arg, *args)
            if isinstance(arg, DirectSumHilbertSpace):
                s = '(%s)' % s
            spaces_strs.append(s)
        return spaces_strs

    def _sympyrepr(self, printer, *args):
        spaces_reprs = self._spaces_printer(printer, *args)
        return 'TensorProductHilbertSpace(%s)' % ','.join(spaces_reprs)

    def _sympystr(self, printer, *args):
        spaces_strs = self._spaces_printer(printer, *args)
        return '*'.join(spaces_strs)

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
                    pform = prettyForm(*pform.right(' ' + '⨂' + ' '))
                else:
                    pform = prettyForm(*pform.right(' x '))
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
                s = s + '\\otimes '
        return s