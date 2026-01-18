import itertools
from sympy.combinatorics.fp_groups import FpGroup, FpSubgroup, simplify_presentation
from sympy.combinatorics.free_groups import FreeGroup
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.numbers import igcd
from sympy.ntheory.factor_ import totient
from sympy.core.singleton import S
def _check_homomorphism(domain, codomain, images):
    """
    Check that a given mapping of generators to images defines a homomorphism.

    Parameters
    ==========
    domain : PermutationGroup, FpGroup, FreeGroup
    codomain : PermutationGroup, FpGroup, FreeGroup
    images : dict
        The set of keys must be equal to domain.generators.
        The values must be elements of the codomain.

    """
    pres = domain if hasattr(domain, 'relators') else domain.presentation()
    rels = pres.relators
    gens = pres.generators
    symbols = [g.ext_rep[0] for g in gens]
    symbols_to_domain_generators = dict(zip(symbols, domain.generators))
    identity = codomain.identity

    def _image(r):
        w = identity
        for symbol, power in r.array_form:
            g = symbols_to_domain_generators[symbol]
            w *= images[g] ** power
        return w
    for r in rels:
        if isinstance(codomain, FpGroup):
            s = codomain.equals(_image(r), identity)
            if s is None:
                success = codomain.make_confluent()
                s = codomain.equals(_image(r), identity)
                if s is None and (not success):
                    raise RuntimeError("Can't determine if the images define a homomorphism. Try increasing the maximum number of rewriting rules (group._rewriting_system.set_max(new_value); the current value is stored in group._rewriting_system.maxeqns)")
        else:
            s = _image(r).is_identity
        if not s:
            return False
    return True