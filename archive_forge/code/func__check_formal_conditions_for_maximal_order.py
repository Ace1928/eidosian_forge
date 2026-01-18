from sympy.polys.polytools import Poly
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
from sympy.utilities.decorator import public
from .basis import round_two, nilradical_mod_p
from .exceptions import StructureError
from .modules import ModuleEndomorphism, find_min_poly
from .utilities import coeff_search, supplement_a_subspace
def _check_formal_conditions_for_maximal_order(submodule):
    """
    Several functions in this module accept an argument which is to be a
    :py:class:`~.Submodule` representing the maximal order in a number field,
    such as returned by the :py:func:`~sympy.polys.numberfields.basis.round_two`
    algorithm.

    We do not attempt to check that the given ``Submodule`` actually represents
    a maximal order, but we do check a basic set of formal conditions that the
    ``Submodule`` must satisfy, at a minimum. The purpose is to catch an
    obviously ill-formed argument.
    """
    prefix = 'The submodule representing the maximal order should '
    cond = None
    if not submodule.is_power_basis_submodule():
        cond = 'be a direct submodule of a power basis.'
    elif not submodule.starts_with_unity():
        cond = 'have 1 as its first generator.'
    elif not submodule.is_sq_maxrank_HNF():
        cond = 'have square matrix, of maximal rank, in Hermite Normal Form.'
    if cond is not None:
        raise StructureError(prefix + cond)