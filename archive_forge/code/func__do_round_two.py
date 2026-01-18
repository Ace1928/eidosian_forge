from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.polyclasses import ANP
from sympy.polys.polyerrors import CoercionFailed, DomainError, NotAlgebraic, IsomorphismFailed
from sympy.utilities import public
def _do_round_two(self):
    from sympy.polys.numberfields.basis import round_two
    ZK, dK = round_two(self, radicals=self._nilradicals_mod_p)
    self._maximal_order = ZK
    self._discriminant = dK