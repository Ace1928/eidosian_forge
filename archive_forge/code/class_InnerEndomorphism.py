from sympy.core.numbers import igcd, ilcm
from sympy.core.symbol import Dummy
from sympy.polys.polyclasses import ANP
from sympy.polys.polytools import Poly
from sympy.polys.densetools import dup_clear_denoms
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMBadInputError
from sympy.polys.matrices.normalforms import hermite_normal_form
from sympy.polys.polyerrors import CoercionFailed, UnificationFailed
from sympy.polys.polyutils import IntegerPowerable
from .exceptions import ClosureFailure, MissingUnityError, StructureError
from .utilities import AlgIntPowers, is_rat, get_num_denom
class InnerEndomorphism(ModuleEndomorphism):
    """
    An inner endomorphism on a module, i.e. the endomorphism corresponding to
    multiplication by a fixed element.
    """

    def __init__(self, domain, multiplier):
        """
        Parameters
        ==========

        domain : :py:class:`~.Module`
            The domain and codomain of the endomorphism.

        multiplier : :py:class:`~.ModuleElement`
            The element $a$ defining the mapping as $x \\mapsto a x$.

        """
        super().__init__(domain, lambda x: multiplier * x)
        self.multiplier = multiplier