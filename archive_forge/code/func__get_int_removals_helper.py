from __future__ import annotations
import math
from collections import defaultdict
import scipy.constants as const
from pymatgen.core import Composition, Element, Species
def _get_int_removals_helper(self, spec_amts_oxi, redox_el, redox_els, num_a):
    """This is a helper method for get_removals_int_oxid!

        Args:
            spec_amts_oxi: a dict of species to their amounts in the structure
            redox_el: the element to oxidize or reduce
            redox_els: the full list of elements that might be oxidized or reduced
            num_a: a running set of numbers of A ion at integer oxidation steps

        Returns:
            a set of numbers A; steps for oxidizing oxid_el first, then the other oxid_els in this list
        """
    if self.working_ion_charge < 0:
        oxid_old = max((spec.oxi_state for spec in spec_amts_oxi if spec.symbol == redox_el.symbol))
        oxid_new = math.ceil(oxid_old - 1)
        lowest_oxid = defaultdict(lambda: 2, {'Cu': 1})
        if oxid_new < min((os for os in Element(redox_el.symbol).oxidation_states if os >= lowest_oxid[redox_el.symbol])):
            return num_a
    else:
        oxid_old = min((spec.oxi_state for spec in spec_amts_oxi if spec.symbol == redox_el.symbol))
        oxid_new = math.floor(oxid_old + 1)
        if oxid_new > redox_el.max_oxidation_state:
            return num_a
    spec_old = Species(redox_el.symbol, oxid_old)
    spec_new = Species(redox_el.symbol, oxid_new)
    spec_amt = spec_amts_oxi[spec_old]
    spec_amts_oxi = {sp: amt for sp, amt in spec_amts_oxi.items() if sp != spec_old}
    spec_amts_oxi[spec_new] = spec_amt
    spec_amts_oxi = Composition(spec_amts_oxi)
    oxi_noA = sum((spec.oxi_state * spec_amts_oxi[spec] for spec in spec_amts_oxi if spec.symbol not in self.working_ion.symbol))
    a = max(0, -oxi_noA / self.working_ion_charge)
    num_a = num_a | {a}
    if a == 0:
        return num_a
    for red in redox_els:
        num_a = num_a | self._get_int_removals_helper(spec_amts_oxi.copy(), red, redox_els, num_a)
    return num_a