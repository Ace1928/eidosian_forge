from sympy.polys.agca.modules import FreeModulePolyRing
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.domains.compositedomain import CompositeDomain
from sympy.polys.domains.old_fractionfield import FractionField
from sympy.polys.domains.ring import Ring
from sympy.polys.orderings import monomial_key, build_product_order
from sympy.polys.polyclasses import DMP, DMF
from sympy.polys.polyerrors import (GeneratorsNeeded, PolynomialError,
from sympy.polys.polyutils import dict_from_basic, basic_from_dict, _dict_reorder
from sympy.utilities import public
from sympy.utilities.iterables import iterable
def _vector_to_sdm_helper(v, order):
    """Helper method for common code in Global and Local poly rings."""
    from sympy.polys.distributedmodules import sdm_from_dict
    d = {}
    for i, e in enumerate(v):
        for key, value in e.to_dict().items():
            d[(i,) + key] = value
    return sdm_from_dict(d, order)