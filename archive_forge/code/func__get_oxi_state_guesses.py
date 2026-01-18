from __future__ import annotations
import collections
import os
import re
import string
import warnings
from functools import total_ordering
from itertools import combinations_with_replacement, product
from math import isnan
from typing import TYPE_CHECKING, cast
from monty.fractions import gcd, gcd_float
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core.periodic_table import DummySpecies, Element, ElementType, Species, get_el_sp
from pymatgen.core.units import Mass
from pymatgen.util.string import Stringify, formula_double_format
def _get_oxi_state_guesses(self, all_oxi_states, max_sites, oxi_states_override, target_charge):
    """Utility operation for guessing oxidation states.

        See `oxi_state_guesses` for full details. This operation does the
        calculation of the most likely oxidation states

        Args:
            oxi_states_override (dict): dict of str->list to override an
                element's common oxidation states, e.g. {"V": [2,3,4,5]}
            target_charge (int): the desired total charge on the structure.
                Default is 0 signifying charge balance.
            all_oxi_states (bool): if True, an element defaults to
                all oxidation states in pymatgen Element.icsd_oxidation_states.
                Otherwise, default is Element.common_oxidation_states. Note
                that the full oxidation state list is *very* inclusive and
                can produce nonsensical results.
            max_sites (int): if possible, will reduce Compositions to at most
                this many sites to speed up oxidation state guesses. If the
                composition cannot be reduced to this many sites a ValueError
                will be raised. Set to -1 to just reduce fully. If set to a
                number less than -1, the formula will be fully reduced but a
                ValueError will be thrown if the number of atoms in the reduced
                formula is greater than abs(max_sites).

        Returns:
            list[dict]: Each dict maps the element symbol to a list of
                oxidation states for each site of that element. For example, Fe3O4 could
                return a list of [2,2,2,3,3,3] for the oxidation states of the 6 Fe sites.
                If the composition is not charge balanced, an empty list is returned.
        """
    comp = self.copy()
    if max_sites and max_sites < 0:
        comp = self.reduced_composition
        if max_sites < -1 and comp.num_atoms > abs(max_sites):
            raise ValueError(f'Composition {comp} cannot accommodate max_sites setting!')
    elif max_sites and comp.num_atoms > max_sites:
        reduced_comp, reduced_factor = self.get_reduced_composition_and_factor()
        if reduced_factor > 1:
            reduced_comp *= max(1, int(max_sites / reduced_comp.num_atoms))
            comp = reduced_comp
        if comp.num_atoms > max_sites:
            raise ValueError(f'Composition {comp} cannot accommodate max_sites setting!')
    if not Composition.oxi_prob:
        all_data = loadfn(f'{module_dir}/../analysis/icsd_bv.yaml')
        Composition.oxi_prob = {Species.from_str(sp): data for sp, data in all_data['occurrence'].items()}
    oxi_states_override = oxi_states_override or {}
    if not all((amt == int(amt) for amt in comp.values())):
        raise ValueError('Charge balance analysis requires integer values in Composition!')
    el_amt = comp.get_el_amt_dict()
    elements = list(el_amt)
    el_sums = []
    el_sum_scores = collections.defaultdict(set)
    el_best_oxid_combo = {}
    for idx, el in enumerate(elements):
        el_sum_scores[idx] = {}
        el_best_oxid_combo[idx] = {}
        el_sums.append([])
        if oxi_states_override.get(el):
            oxids = oxi_states_override[el]
        elif all_oxi_states:
            oxids = Element(el).oxidation_states
        else:
            oxids = Element(el).icsd_oxidation_states or Element(el).oxidation_states
        for oxid_combo in combinations_with_replacement(oxids, int(el_amt[el])):
            oxid_sum = sum(oxid_combo)
            if oxid_sum not in el_sums[idx]:
                el_sums[idx].append(oxid_sum)
            score = sum((Composition.oxi_prob.get(Species(el, o), 0) for o in oxid_combo))
            if oxid_sum not in el_sum_scores[idx] or score > el_sum_scores[idx].get(oxid_sum, 0):
                el_sum_scores[idx][oxid_sum] = score
                el_best_oxid_combo[idx][oxid_sum] = oxid_combo
    all_sols = []
    all_oxid_combo = []
    all_scores = []
    for x in product(*el_sums):
        if sum(x) == target_charge:
            el_sum_sol = dict(zip(elements, x))
            sol = {el: v / el_amt[el] for el, v in el_sum_sol.items()}
            all_sols.append(sol)
            score = 0
            for idx, v in enumerate(x):
                score += el_sum_scores[idx][v]
            all_scores.append(score)
            all_oxid_combo.append({e: el_best_oxid_combo[idx][v] for idx, (e, v) in enumerate(zip(elements, x))})
    if all_scores:
        all_sols, all_oxid_combo = zip(*((y, x) for z, y, x in sorted(zip(all_scores, all_sols, all_oxid_combo), key=lambda pair: pair[0], reverse=True)))
    return (tuple(all_sols), tuple(all_oxid_combo))