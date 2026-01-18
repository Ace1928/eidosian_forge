from __future__ import annotations
import collections
import functools
import operator
import os
from math import exp, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.serialization import loadfn
from pymatgen.core import Element, Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class BVAnalyzer:
    """
    This class implements a maximum a posteriori (MAP) estimation method to
    determine oxidation states in a structure. The algorithm is as follows:
    1) The bond valence sum of all symmetrically distinct sites in a structure
    is calculated using the element-based parameters in M. O'Keefe, & N. Brese,
    JACS, 1991, 113(9), 3226-3229. doi:10.1021/ja00009a002.
    2) The posterior probabilities of all oxidation states is then calculated
    using: P(oxi_state/BV) = K * P(BV/oxi_state) * P(oxi_state), where K is
    a constant factor for each element. P(BV/oxi_state) is calculated as a
    Gaussian with mean and std deviation determined from an analysis of
    the ICSD. The posterior P(oxi_state) is determined from a frequency
    analysis of the ICSD.
    3) The oxidation states are then ranked in order of decreasing probability
    and the oxidation state combination that result in a charge neutral cell
    is selected.
    """
    CHARGE_NEUTRALITY_TOLERANCE = 1e-05

    def __init__(self, symm_tol=0.1, max_radius=4, max_permutations=100000, distance_scale_factor=1.015, charge_neutrality_tolerance=CHARGE_NEUTRALITY_TOLERANCE, forbidden_species=None):
        """
        Initializes the BV analyzer, with useful defaults.

        Args:
            symm_tol:
                Symmetry tolerance used to determine which sites are
                symmetrically equivalent. Set to 0 to turn off symmetry.
            max_radius:
                Maximum radius in Angstrom used to find nearest neighbors.
            max_permutations:
                The maximum number of permutations of oxidation states to test.
            distance_scale_factor:
                A scale factor to be applied. This is useful for scaling
                distances, esp in the case of calculation-relaxed structures
                which may tend to under (GGA) or over bind (LDA). The default
                of 1.015 works for GGA. For experimental structure, set this to
                1.
            charge_neutrality_tolerance:
                Tolerance on the charge neutrality when unordered structures
                are at stake.
            forbidden_species:
                List of species that are forbidden (example : ["O-"] cannot be
                used) It is used when e.g. someone knows that some oxidation
                state cannot occur for some atom in a structure or list of
                structures.
        """
        self.symm_tol = symm_tol
        self.max_radius = max_radius
        self.max_permutations = max_permutations
        self.dist_scale_factor = distance_scale_factor
        self.charge_neutrality_tolerance = charge_neutrality_tolerance
        forbidden_species = [get_el_sp(sp) for sp in forbidden_species] if forbidden_species else []
        self.icsd_bv_data = {get_el_sp(specie): data for specie, data in ICSD_BV_DATA.items() if specie not in forbidden_species} if len(forbidden_species) > 0 else ICSD_BV_DATA

    def _calc_site_probabilities(self, site, nn):
        el = site.specie.symbol
        bv_sum = calculate_bv_sum(site, nn, scale_factor=self.dist_scale_factor)
        prob = {}
        for sp, data in self.icsd_bv_data.items():
            if sp.symbol == el and sp.oxi_state != 0 and (data['std'] > 0):
                u = data['mean']
                sigma = data['std']
                prob[sp.oxi_state] = exp(-(bv_sum - u) ** 2 / 2 / sigma ** 2) / sigma * PRIOR_PROB[sp]
        try:
            prob = {k: v / sum(prob.values()) for k, v in prob.items()}
        except ZeroDivisionError:
            prob = dict.fromkeys(prob, 0)
        return prob

    def _calc_site_probabilities_unordered(self, site, nn):
        bv_sum = calculate_bv_sum_unordered(site, nn, scale_factor=self.dist_scale_factor)
        prob = {}
        for specie in site.species:
            el = specie.symbol
            prob[el] = {}
            for sp, data in self.icsd_bv_data.items():
                if sp.symbol == el and sp.oxi_state != 0 and (data['std'] > 0):
                    u = data['mean']
                    sigma = data['std']
                    prob[el][sp.oxi_state] = exp(-(bv_sum - u) ** 2 / 2 / sigma ** 2) / sigma * PRIOR_PROB[sp]
            try:
                prob[el] = {k: v / sum(prob[el].values()) for k, v in prob[el].items()}
            except ZeroDivisionError:
                prob[el] = dict.fromkeys(prob[el], 0)
        return prob

    def get_valences(self, structure: Structure):
        """
        Returns a list of valences for each site in the structure.

        Args:
            structure: Structure to analyze

        Returns:
            A list of valences for each site in the structure (for an ordered structure),
            e.g., [1, 1, -2] or a list of lists with the valences for each fractional
            element of each site in the structure (for an unordered structure), e.g., [[2,
            4], [3], [-2], [-2], [-2]]

        Raises:
            A ValueError if the valences cannot be determined.
        """
        els = [Element(el.symbol) for el in structure.elements]
        if (diff := (set(els) - set(BV_PARAMS))):
            raise ValueError(f'Structure contains elements not in set of BV parameters: {diff}')
        if self.symm_tol:
            finder = SpacegroupAnalyzer(structure, self.symm_tol)
            symm_structure = finder.get_symmetrized_structure()
            equi_sites = symm_structure.equivalent_sites
        else:
            equi_sites = [[site] for site in structure]
        equi_sites = sorted(equi_sites, key=lambda sites: -sites[0].species.average_electroneg)
        valences = []
        all_prob = []
        if structure.is_ordered:
            for sites in equi_sites:
                test_site = sites[0]
                nn = structure.get_neighbors(test_site, self.max_radius)
                prob = self._calc_site_probabilities(test_site, nn)
                all_prob.append(prob)
                val = list(prob)
                val = sorted(val, key=lambda v: -prob[v])
                valences.append(list(filter(lambda v: prob[v] > 0.01 * prob[val[0]], val)))
        else:
            full_all_prob = []
            for sites in equi_sites:
                test_site = sites[0]
                nn = structure.get_neighbors(test_site, self.max_radius)
                prob = self._calc_site_probabilities_unordered(test_site, nn)
                all_prob.append(prob)
                full_all_prob.extend(prob.values())
                vals = []
                for elem, _ in get_z_ordered_elmap(test_site.species):
                    val = list(prob[elem.symbol])
                    val = sorted(val, key=lambda v: -prob[elem.symbol][v])
                    filtered = list(filter(lambda v: prob[elem.symbol][v] > 0.001 * prob[elem.symbol][val[0]], val))
                    vals.append(filtered)
                valences.append(vals)
        if structure.is_ordered:
            n_sites = np.array(list(map(len, equi_sites)))
            valence_min = np.array(list(map(min, valences)))
            valence_max = np.array(list(map(max, valences)))
            self._n = 0
            self._best_score = 0
            self._best_vset = None

            def evaluate_assignment(v_set):
                el_oxi = collections.defaultdict(list)
                for idx, sites in enumerate(equi_sites):
                    el_oxi[sites[0].specie.symbol].append(v_set[idx])
                max_diff = max((max(v) - min(v) for v in el_oxi.values()))
                if max_diff > 1:
                    return
                score = functools.reduce(operator.mul, [all_prob[idx][val] for idx, val in enumerate(v_set)])
                if score > self._best_score:
                    self._best_vset = v_set
                    self._best_score = score

            def _recurse(assigned=None):
                if self._n > self.max_permutations:
                    return
                if assigned is None:
                    assigned = []
                i = len(assigned)
                highest = valence_max.copy()
                highest[:i] = assigned
                highest *= n_sites
                highest = np.sum(highest)
                lowest = valence_min.copy()
                lowest[:i] = assigned
                lowest *= n_sites
                lowest = np.sum(lowest)
                if highest < 0 or lowest > 0:
                    self._n += 1
                    return
                if i == len(valences):
                    evaluate_assignment(assigned)
                    self._n += 1
                    return
                for v in valences[i]:
                    new_assigned = list(assigned)
                    _recurse([*new_assigned, v])
                return
        else:
            n_sites = np.array([len(sites) for sites in equi_sites])
            tmp = []
            attrib = []
            for idx, n_site in enumerate(n_sites):
                for _ in valences[idx]:
                    tmp.append(n_site)
                    attrib.append(idx)
            new_n_sites = np.array(tmp)
            fractions = []
            elements = []
            for sites in equi_sites:
                for sp, occu in get_z_ordered_elmap(sites[0].species):
                    elements.append(sp.symbol)
                    fractions.append(occu)
            fractions = np.array(fractions, float)
            new_valences = [val for vals in valences for val in vals]
            valence_min = np.array([min(val) for val in new_valences], float)
            valence_max = np.array([max(val) for val in new_valences], float)
            self._n = 0
            self._best_score = 0
            self._best_vset = None

            def evaluate_assignment(v_set):
                el_oxi = collections.defaultdict(list)
                jj = 0
                for sites in equi_sites:
                    for specie, _ in get_z_ordered_elmap(sites[0].species):
                        el_oxi[specie.symbol].append(v_set[jj])
                        jj += 1
                max_diff = max((max(v) - min(v) for v in el_oxi.values()))
                if max_diff > 2:
                    return
                score = functools.reduce(operator.mul, [all_prob[attrib[iv]][elements[iv]][vv] for iv, vv in enumerate(v_set)])
                if score > self._best_score:
                    self._best_vset = v_set
                    self._best_score = score

            def _recurse(assigned=None):
                if self._n > self.max_permutations:
                    return
                if assigned is None:
                    assigned = []
                i = len(assigned)
                highest = valence_max.copy()
                highest[:i] = assigned
                highest *= new_n_sites
                highest *= fractions
                highest = np.sum(highest)
                lowest = valence_min.copy()
                lowest[:i] = assigned
                lowest *= new_n_sites
                lowest *= fractions
                lowest = np.sum(lowest)
                if highest < -self.charge_neutrality_tolerance or lowest > self.charge_neutrality_tolerance:
                    self._n += 1
                    return
                if i == len(new_valences):
                    evaluate_assignment(assigned)
                    self._n += 1
                    return
                for v in new_valences[i]:
                    new_assigned = list(assigned)
                    _recurse([*new_assigned, v])
                return
        _recurse()
        if self._best_vset:
            if structure.is_ordered:
                assigned = {}
                for val, sites in zip(self._best_vset, equi_sites):
                    for site in sites:
                        assigned[site] = val
                return [int(assigned[site]) for site in structure]
            assigned = {}
            new_best_vset = []
            for _ in equi_sites:
                new_best_vset.append([])
            for ival, val in enumerate(self._best_vset):
                new_best_vset[attrib[ival]].append(val)
            for val, sites in zip(new_best_vset, equi_sites):
                for site in sites:
                    assigned[site] = val
            return [[int(frac_site) for frac_site in assigned[site]] for site in structure]
        raise ValueError('Valences cannot be assigned!')

    def get_oxi_state_decorated_structure(self, structure: Structure):
        """
        Get an oxidation state decorated structure. This currently works only
        for ordered structures only.

        Args:
            structure: Structure to analyze

        Returns:
            A modified structure that is oxidation state decorated.

        Raises:
            ValueError if the valences cannot be determined.
        """
        struct = structure.copy()
        if struct.is_ordered:
            valences = self.get_valences(struct)
            struct.add_oxidation_state_by_site(valences)
        else:
            valences = self.get_valences(struct)
            struct = add_oxidation_state_by_site_fraction(struct, valences)
        return struct