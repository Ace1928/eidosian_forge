import sympy
from sympy.multipledispatch import dispatch
class SingletonInt(sympy.AtomicExpr):
    _op_priority = 99999

    def __new__(cls, *args, coeff=None, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        return instance

    def __init__(self, val, *, coeff=1):
        self._val = val
        self._coeff = coeff
        super().__init__()

    def _eval_Eq(self, other):
        if isinstance(other, SingletonInt) and other._val == self._val and (self._coeff == other._coeff):
            return sympy.true
        else:
            return sympy.false

    @property
    def free_symbols(self):
        return set()

    def __mul__(self, other):
        if isinstance(other, SingletonInt):
            raise ValueError('SingletonInt cannot be multiplied by another SingletonInt')
        return SingletonInt(self._val, coeff=self._coeff * other)

    def __rmul__(self, other):
        if isinstance(other, SingletonInt):
            raise ValueError('SingletonInt cannot be multiplied by another SingletonInt')
        return SingletonInt(self._val, coeff=self._coeff * other)

    def __add__(self, other):
        raise NotImplementedError('NYI')

    def __sub__(self, other):
        raise NotImplementedError('NYI')

    def __truediv__(self, other):
        raise NotImplementedError('NYI')

    def __floordiv__(self, other):
        raise NotImplementedError('NYI')

    def __mod__(self, other):
        raise NotImplementedError('NYI')