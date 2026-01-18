from sympy.polys.domains.domainelement import DomainElement
from sympy.utilities import public
from mpmath.ctx_mp_python import PythonMPContext, _mpf, _mpc, _constant
from mpmath.libmp import (MPZ_ONE, fzero, fone, finf, fninf, fnan,
from mpmath.rational import mpq
@public
class ComplexElement(_mpc, DomainElement):
    """An element of a complex domain. """
    __slots__ = ('__mpc__',)

    def _set_mpc(self, val):
        self.__mpc__ = val
    _mpc_ = property(lambda self: self.__mpc__, _set_mpc)

    def parent(self):
        return self.context._parent