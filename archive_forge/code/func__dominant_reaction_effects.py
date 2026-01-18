from .. import ReactionSystem, Equilibrium
from ..units import get_derived_unit, to_unitless, default_units as u
def _dominant_reaction_effects(substance_key, rsys, rates, linthreshy, eqk1, eqk2, eqs):
    tot = np.zeros(rates.shape[0])
    reaction_effects = rsys.per_reaction_effect_on_substance(substance_key)
    data = []
    for ri, n in reaction_effects.items():
        tot += n * rates[..., ri]
        if ri in eqk1:
            otheri = eqk2[eqk1.index(ri)]
            y = n * rates[..., ri] + reaction_effects[otheri] * rates[..., otheri]
            rxn = eqs[eqk1.index(ri)]
        elif ri in eqk2:
            continue
        else:
            y = n * rates[..., ri]
            rxn = rsys.rxns[ri]
        if np.all(np.abs(y) < linthreshy):
            continue
        data.append((y, rxn))
    return (data, tot)