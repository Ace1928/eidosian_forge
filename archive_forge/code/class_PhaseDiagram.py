from __future__ import annotations
import collections
import itertools
import json
import logging
import math
import os
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, no_type_check
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from monty.json import MontyDecoder, MSONable
from scipy import interpolate
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from tqdm import tqdm
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import DummySpecies, Element, get_el_sp
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.coord import Simplex, in_coord_list
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
@due.dcite(Doi('10.1021/cm702327g'), description='Phase Diagram from First Principles Calculations')
@due.dcite(Doi('10.1016/j.elecom.2010.01.010'), description='Thermal stabilities of delithiated olivine MPO4 (M=Fe, Mn) cathodes investigated using first principles calculations')
class PhaseDiagram(MSONable):
    """
    Simple phase diagram class taking in elements and entries as inputs.
    The algorithm is based on the work in the following papers:

    1. S. P. Ong, L. Wang, B. Kang, and G. Ceder, Li-Fe-P-O2 Phase Diagram from
        First Principles Calculations. Chem. Mater., 2008, 20(5), 1798-1807.
        doi:10.1021/cm702327g

    2. S. P. Ong, A. Jain, G. Hautier, B. Kang, G. Ceder, Thermal stabilities
        of delithiated olivine MPO4 (M=Fe, Mn) cathodes investigated using first
        principles calculations. Electrochem. Comm., 2010, 12(3), 427-430.
        doi:10.1016/j.elecom.2010.01.010

    Attributes:
        dim (int): The dimensionality of the phase diagram.
        elements: Elements in the phase diagram.
        el_refs: List of elemental references for the phase diagrams. These are
            entries corresponding to the lowest energy element entries for simple
            compositional phase diagrams.
        all_entries: All entries provided for Phase Diagram construction. Note that this
            does not mean that all these entries are actually used in the phase
            diagram. For example, this includes the positive formation energy
            entries that are filtered out before Phase Diagram construction.
        qhull_entries: Actual entries used in convex hull. Excludes all positive formation
            energy entries.
        qhull_data: Data used in the convex hull operation. This is essentially a matrix of
            composition data and energy per atom values created from qhull_entries.
        facets: Facets of the phase diagram in the form of  [[1,2,3],[4,5,6]...].
            For a ternary, it is the indices (references to qhull_entries and
            qhull_data) for the vertices of the phase triangles. Similarly
            extended to higher D simplices for higher dimensions.
        simplices: The simplices of the phase diagram as a list of np.ndarray, i.e.,
            the list of stable compositional coordinates in the phase diagram.
    """
    formation_energy_tol = 1e-11
    numerical_tol = 1e-08

    def __init__(self, entries: Sequence[PDEntry] | set[PDEntry], elements: Sequence[Element]=(), *, computed_data: dict[str, Any] | None=None) -> None:
        """
        Args:
            entries (list[PDEntry]): A list of PDEntry-like objects having an
                energy, energy_per_atom and composition.
            elements (list[Element]): Optional list of elements in the phase
                diagram. If set to None, the elements are determined from
                the entries themselves and are sorted alphabetically.
                If specified, element ordering (e.g. for pd coordinates)
                is preserved.
            computed_data (dict): A dict containing pre-computed data. This allows
                PhaseDiagram object to be reconstituted without performing the
                expensive convex hull computation. The dict is the output from the
                PhaseDiagram._compute() method and is stored in PhaseDiagram.computed_data
                when generated for the first time.
        """
        if not entries:
            raise ValueError('Unable to build phase diagram without entries.')
        self.elements = elements
        self.entries = entries
        if computed_data is None:
            computed_data = self._compute()
        else:
            computed_data = MontyDecoder().process_decoded(computed_data)
            assert isinstance(computed_data, dict)
            computed_data['el_refs'] = [(Element(el_str), entry) for el_str, entry in computed_data['el_refs']]
        self.computed_data = computed_data
        self.facets = computed_data['facets']
        self.simplexes = computed_data['simplexes']
        self.all_entries = computed_data['all_entries']
        self.qhull_data = computed_data['qhull_data']
        self.dim = computed_data['dim']
        self.el_refs = dict(computed_data['el_refs'])
        self.qhull_entries = tuple(computed_data['qhull_entries'])
        self._qhull_spaces = tuple((frozenset(e.elements) for e in self.qhull_entries))
        self._stable_entries = tuple({self.qhull_entries[idx] for idx in set(itertools.chain(*self.facets))})
        self._stable_spaces = tuple((frozenset(e.elements) for e in self._stable_entries))

    def as_dict(self):
        """
        Returns:
            MSONable dictionary representation of PhaseDiagram.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'all_entries': [e.as_dict() for e in self.all_entries], 'elements': [e.as_dict() for e in self.elements], 'computed_data': self.computed_data}

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> Self:
        """
        Args:
            dct (dict): dictionary representation of PhaseDiagram.

        Returns:
            PhaseDiagram
        """
        entries = [MontyDecoder().process_decoded(entry) for entry in dct['all_entries']]
        elements = [Element.from_dict(elem) for elem in dct['elements']]
        computed_data = dct.get('computed_data')
        return cls(entries, elements, computed_data=computed_data)

    def _compute(self) -> dict[str, Any]:
        if self.elements == ():
            self.elements = sorted({els for e in self.entries for els in e.elements})
        elements = list(self.elements)
        dim = len(elements)
        entries = sorted(self.entries, key=lambda e: e.composition.reduced_composition)
        el_refs: dict[Element, PDEntry] = {}
        min_entries: list[PDEntry] = []
        all_entries: list[PDEntry] = []
        for composition, group_iter in itertools.groupby(entries, key=lambda e: e.composition.reduced_composition):
            group = list(group_iter)
            min_entry = min(group, key=lambda e: e.energy_per_atom)
            if composition.is_element:
                el_refs[composition.elements[0]] = min_entry
            min_entries.append(min_entry)
            all_entries.extend(group)
        if (missing := (set(elements) - set(el_refs))):
            raise ValueError(f'Missing terminal entries for elements {sorted(map(str, missing))}')
        if (extra := (set(el_refs) - set(elements))):
            raise ValueError(f'There are more terminal elements than dimensions: {sorted(map(str, extra))}')
        data = np.array([[e.composition.get_atomic_fraction(el) for el in elements] + [e.energy_per_atom] for e in min_entries])
        vec = [el_refs[el].energy_per_atom for el in elements] + [-1]
        form_e = -np.dot(data, vec)
        idx = np.where(form_e < -PhaseDiagram.formation_energy_tol)[0].tolist()
        idx.extend([min_entries.index(el) for el in el_refs.values()])
        qhull_entries = [min_entries[idx] for idx in idx]
        qhull_data = data[idx][:, 1:]
        extra_point = np.zeros(dim) + 1 / dim
        extra_point[-1] = np.max(qhull_data) + 1
        qhull_data = np.concatenate([qhull_data, [extra_point]], axis=0)
        if dim == 1:
            facets = [qhull_data.argmin(axis=0)]
        else:
            facets = get_facets(qhull_data)
            final_facets = []
            for facet in facets:
                if max(facet) == len(qhull_data) - 1:
                    continue
                mat = qhull_data[facet]
                mat[:, -1] = 1
                if abs(np.linalg.det(mat)) > 1e-14:
                    final_facets.append(facet)
            facets = final_facets
        simplexes = [Simplex(qhull_data[facet, :-1]) for facet in facets]
        self.elements = elements
        return {'facets': facets, 'simplexes': simplexes, 'all_entries': all_entries, 'qhull_data': qhull_data, 'dim': dim, 'el_refs': list(el_refs.items()), 'qhull_entries': qhull_entries}

    def pd_coords(self, comp: Composition) -> np.ndarray:
        """
        The phase diagram is generated in a reduced dimensional space
        (n_elements - 1). This function returns the coordinates in that space.
        These coordinates are compatible with the stored simplex objects.

        Args:
            comp (Composition): A composition

        Returns:
            The coordinates for a given composition in the PhaseDiagram's basis
        """
        if set(comp.elements) - set(self.elements):
            raise ValueError(f'{comp} has elements not in the phase diagram {', '.join(map(str, self.elements))}')
        return np.array([comp.get_atomic_fraction(el) for el in self.elements[1:]])

    @property
    def all_entries_hulldata(self):
        """
        Returns:
            The actual ndarray used to construct the convex hull.
        """
        data = [[e.composition.get_atomic_fraction(el) for el in self.elements] + [e.energy_per_atom] for e in self.all_entries]
        return np.array(data)[:, 1:]

    @property
    def unstable_entries(self) -> set[Entry]:
        """
        Returns:
            set[Entry]: unstable entries in the phase diagram. Includes positive formation energy entries.
        """
        return {e for e in self.all_entries if e not in self.stable_entries}

    @property
    def stable_entries(self) -> set[Entry]:
        """
        Returns:
            set[Entry]: of stable entries in the phase diagram.
        """
        return set(self._stable_entries)

    @lru_cache(1)
    def _get_stable_entries_in_space(self, space) -> list[Entry]:
        """
        Args:
            space (set[Element]): set of Element objects.

        Returns:
            list[Entry]: stable entries in the space.
        """
        return [e for e, s in zip(self._stable_entries, self._stable_spaces) if space.issuperset(s)]

    def get_reference_energy(self, comp: Composition) -> float:
        """Sum of elemental reference energies over all elements in a composition.

        Args:
            comp (Composition): Input composition.

        Returns:
            float: Reference energy
        """
        return sum((comp[el] * self.el_refs[el].energy_per_atom for el in comp.elements))

    def get_reference_energy_per_atom(self, comp: Composition) -> float:
        """Sum of elemental reference energies over all elements in a composition.

        Args:
            comp (Composition): Input composition.

        Returns:
            float: Reference energy per atom
        """
        return self.get_reference_energy(comp) / comp.num_atoms

    def get_form_energy(self, entry: PDEntry) -> float:
        """
        Returns the formation energy for an entry (NOT normalized) from the
        elemental references.

        Args:
            entry (PDEntry): A PDEntry-like object.

        Returns:
            float: Formation energy from the elemental references.
        """
        comp = entry.composition
        return entry.energy - self.get_reference_energy(comp)

    def get_form_energy_per_atom(self, entry: PDEntry) -> float:
        """
        Returns the formation energy per atom for an entry from the
        elemental references.

        Args:
            entry (PDEntry): An PDEntry-like object

        Returns:
            Formation energy **per atom** from the elemental references.
        """
        return self.get_form_energy(entry) / entry.composition.num_atoms

    def __repr__(self) -> str:
        symbols = [el.symbol for el in self.elements]
        output = [f'{'-'.join(symbols)} phase diagram', f'{len(self.stable_entries)} stable phases: ', ', '.join((entry.name for entry in sorted(self.stable_entries, key=str)))]
        return '\n'.join(output)

    @lru_cache(1)
    def _get_facet_and_simplex(self, comp: Composition) -> tuple[Simplex, Simplex]:
        """
        Get any facet that a composition falls into. Cached so successive
        calls at same composition are fast.

        Args:
            comp (Composition): A composition
        """
        coord = self.pd_coords(comp)
        for facet, simplex in zip(self.facets, self.simplexes):
            if simplex.in_simplex(coord, PhaseDiagram.numerical_tol / 10):
                return (facet, simplex)
        raise RuntimeError(f'No facet found for comp = {comp!r}')

    def _get_all_facets_and_simplexes(self, comp):
        """
        Get all facets that a composition falls into.

        Args:
            comp (Composition): A composition
        """
        coords = self.pd_coords(comp)
        all_facets = [facet for facet, simplex in zip(self.facets, self.simplexes) if simplex.in_simplex(coords, PhaseDiagram.numerical_tol / 10)]
        if not all_facets:
            raise RuntimeError(f'No facets found for comp = {comp!r}')
        return all_facets

    def _get_facet_chempots(self, facet):
        """
        Calculates the chemical potentials for each element within a facet.

        Args:
            facet: Facet of the phase diagram.

        Returns:
            {element: chempot} for all elements in the phase diagram.
        """
        comp_list = [self.qhull_entries[idx].composition for idx in facet]
        energy_list = [self.qhull_entries[idx].energy_per_atom for idx in facet]
        atom_frac_mat = [[c.get_atomic_fraction(e) for e in self.elements] for c in comp_list]
        chempots = np.linalg.solve(atom_frac_mat, energy_list)
        return dict(zip(self.elements, chempots))

    def _get_simplex_intersections(self, c1, c2):
        """
        Returns coordinates of the intersection of the tie line between two compositions
        and the simplexes of the PhaseDiagram.

        Args:
            c1: Reduced dimension coordinates of first composition
            c2: Reduced dimension coordinates of second composition

        Returns:
            Array of the intersections between the tie line and the simplexes of
            the PhaseDiagram
        """
        intersections = [c1, c2]
        for sc in self.simplexes:
            intersections.extend(sc.line_intersection(c1, c2))
        return np.array(intersections)

    def get_decomposition(self, comp: Composition) -> dict[PDEntry, float]:
        """
        Provides the decomposition at a particular composition.

        Args:
            comp (Composition): A composition

        Returns:
            Decomposition as a dict of {PDEntry: amount} where amount
            is the amount of the fractional composition.
        """
        facet, simplex = self._get_facet_and_simplex(comp)
        decomp_amts = simplex.bary_coords(self.pd_coords(comp))
        return {self.qhull_entries[f]: amt for f, amt in zip(facet, decomp_amts) if abs(amt) > PhaseDiagram.numerical_tol}

    def get_decomp_and_hull_energy_per_atom(self, comp: Composition) -> tuple[dict[PDEntry, float], float]:
        """
        Args:
            comp (Composition): Input composition.

        Returns:
            Energy of lowest energy equilibrium at desired composition per atom
        """
        decomp = self.get_decomposition(comp)
        return (decomp, sum((e.energy_per_atom * n for e, n in decomp.items())))

    def get_hull_energy_per_atom(self, comp: Composition, **kwargs) -> float:
        """
        Args:
            comp (Composition): Input composition.

        Returns:
            Energy of lowest energy equilibrium at desired composition.
        """
        return self.get_decomp_and_hull_energy_per_atom(comp, **kwargs)[1]

    def get_hull_energy(self, comp: Composition) -> float:
        """
        Args:
            comp (Composition): Input composition.

        Returns:
            Energy of lowest energy equilibrium at desired composition. Not
                normalized by atoms, i.e. E(Li4O2) = 2 * E(Li2O)
        """
        return comp.num_atoms * self.get_hull_energy_per_atom(comp)

    def get_decomp_and_e_above_hull(self, entry: PDEntry, allow_negative: bool=False, check_stable: bool=True, on_error: Literal['raise', 'warn', 'ignore']='raise') -> tuple[dict[PDEntry, float], float] | tuple[None, None]:
        """
        Provides the decomposition and energy above convex hull for an entry.
        Due to caching, can be much faster if entries with the same composition
        are processed together.

        Args:
            entry (PDEntry): A PDEntry like object
            allow_negative (bool): Whether to allow negative e_above_hulls. Used to
                calculate equilibrium reaction energies. Defaults to False.
            check_stable (bool): Whether to first check whether an entry is stable.
                In normal circumstances, this is the faster option since checking for
                stable entries is relatively fast. However, if you have a huge proportion
                of unstable entries, then this check can slow things down. You should then
                set this to False.
            on_error ('raise' | 'warn' | 'ignore'): What to do if no valid decomposition was
                found. 'raise' will throw ValueError. 'warn' will print return (None, None).
                'ignore' just returns (None, None). Defaults to 'raise'.

        Raises:
            ValueError: If on_error is 'raise' and no valid decomposition exists in this
                phase diagram for given entry.

        Returns:
            tuple[decomp, energy_above_hull]: The decomposition is provided
                as a dict of {PDEntry: amount} where amount is the amount of the
                fractional composition. Stable entries should have energy above
                convex hull of 0. The energy is given per atom.
        """
        if check_stable and entry in self.stable_entries:
            return ({entry: 1.0}, 0.0)
        try:
            decomp, hull_energy = self.get_decomp_and_hull_energy_per_atom(entry.composition)
        except Exception as exc:
            if on_error == 'raise':
                raise ValueError(f'Unable to get decomposition for {entry}') from exc
            if on_error == 'warn':
                warnings.warn(f'Unable to get decomposition for {entry}, encountered {exc}')
            return (None, None)
        e_above_hull = entry.energy_per_atom - hull_energy
        if allow_negative or e_above_hull >= -PhaseDiagram.numerical_tol:
            return (decomp, e_above_hull)
        msg = f'No valid decomposition found for {entry}! (e_h: {e_above_hull})'
        if on_error == 'raise':
            raise ValueError(msg)
        if on_error == 'warn':
            warnings.warn(msg)
        return (None, None)

    def get_e_above_hull(self, entry: PDEntry, **kwargs: Any) -> float | None:
        """
        Provides the energy above convex hull for an entry.

        Args:
            entry (PDEntry): A PDEntry like object.
            **kwargs: Passed to get_decomp_and_e_above_hull().

        Returns:
            float | None: Energy above convex hull of entry. Stable entries should have
                energy above hull of 0. The energy is given per atom.
        """
        return self.get_decomp_and_e_above_hull(entry, **kwargs)[1]

    def get_equilibrium_reaction_energy(self, entry: PDEntry) -> float | None:
        """
        Provides the reaction energy of a stable entry from the neighboring
        equilibrium stable entries (also known as the inverse distance to
        hull).

        Args:
            entry (PDEntry): A PDEntry like object

        Returns:
            float | None: Equilibrium reaction energy of entry. Stable entries should have
                equilibrium reaction energy <= 0. The energy is given per atom.
        """
        elem_space = entry.elements
        if entry not in self._get_stable_entries_in_space(frozenset(elem_space)):
            raise ValueError(f'{entry} is unstable, the equilibrium reaction energy is available only for stable entries.')
        if entry.is_element:
            return 0
        entries = [e for e in self._get_stable_entries_in_space(frozenset(elem_space)) if e != entry]
        mod_pd = PhaseDiagram(entries, elements=elem_space)
        return mod_pd.get_decomp_and_e_above_hull(entry, allow_negative=True)[1]

    def get_decomp_and_phase_separation_energy(self, entry: PDEntry, space_limit: int=200, stable_only: bool=False, tols: Sequence[float]=(1e-08,), maxiter: int=1000, **kwargs: Any) -> tuple[dict[PDEntry, float], float] | tuple[None, None]:
        """
        Provides the combination of entries in the PhaseDiagram that gives the
        lowest formation enthalpy with the same composition as the given entry
        excluding entries with the same composition and the energy difference
        per atom between the given entry and the energy of the combination found.

        For unstable entries that are not polymorphs of stable entries (or completely
        novel entries) this is simply the energy above (or below) the convex hull.

        For entries with the same composition as one of the stable entries in the
        phase diagram setting `stable_only` to `False` (Default) allows for entries
        not previously on the convex hull to be considered in the combination.
        In this case the energy returned is what is referred to as the decomposition
        enthalpy in:

        1. Bartel, C., Trewartha, A., Wang, Q., Dunn, A., Jain, A., Ceder, G.,
            A critical examination of compound stability predictions from
            machine-learned formation energies, npj Computational Materials 6, 97 (2020)

        For stable entries setting `stable_only` to `True` returns the same energy
        as `get_equilibrium_reaction_energy`. This function is based on a constrained
        optimization rather than recalculation of the convex hull making it
        algorithmically cheaper. However, if `tol` is too loose there is potential
        for this algorithm to converge to a different solution.

        Args:
            entry (PDEntry): A PDEntry like object.
            space_limit (int): The maximum number of competing entries to consider
                before calculating a second convex hull to reducing the complexity
                of the optimization.
            stable_only (bool): Only use stable materials as competing entries.
            tols (list[float]): Tolerances for convergence of the SLSQP optimization
                when finding the equilibrium reaction. Tighter tolerances tested first.
            maxiter (int): The maximum number of iterations of the SLSQP optimizer
                when finding the equilibrium reaction.
            **kwargs: Passed to get_decomp_and_e_above_hull.

        Returns:
            tuple[decomp, energy]: The decomposition  is given as a dict of {PDEntry, amount}
                for all entries in the decomp reaction where amount is the amount of the
                fractional composition. The phase separation energy is given per atom.
        """
        entry_frac = entry.composition.fractional_composition
        entry_elems = frozenset(entry_frac.elements)
        if entry.is_element:
            return self.get_decomp_and_e_above_hull(entry, allow_negative=True, **kwargs)
        if stable_only:
            compare_entries = self._get_stable_entries_in_space(entry_elems)
        else:
            compare_entries = [e for e, s in zip(self.qhull_entries, self._qhull_spaces) if entry_elems.issuperset(s)]
        same_comp_mem_ids = [id(c) for c in compare_entries if len(entry_frac) == len(c.composition) and all((abs(v - c.composition.get_atomic_fraction(el)) <= Composition.amount_tolerance for el, v in entry_frac.items()))]
        if not any((id(e) in same_comp_mem_ids for e in self._get_stable_entries_in_space(entry_elems))):
            return self.get_decomp_and_e_above_hull(entry, allow_negative=True, **kwargs)
        competing_entries = {c for c in compare_entries if id(c) not in same_comp_mem_ids}
        if len(competing_entries) > space_limit and (not stable_only):
            warnings.warn(f'There are {len(competing_entries)} competing entries for {entry.composition} - Calculating inner hull to discard additional unstable entries')
            reduced_space = competing_entries - {*self._get_stable_entries_in_space(entry_elems)} | {*self.el_refs.values()}
            inner_hull = PhaseDiagram(reduced_space)
            competing_entries = inner_hull.stable_entries | {*self._get_stable_entries_in_space(entry_elems)}
            competing_entries = {c for c in compare_entries if id(c) not in same_comp_mem_ids}
        if len(competing_entries) > space_limit:
            warnings.warn(f'There are {len(competing_entries)} competing entries for {entry.composition} - Using SLSQP to find decomposition likely to be slow')
        decomp = _get_slsqp_decomp(entry.composition, competing_entries, tols, maxiter)
        decomp_enthalpy = np.sum([c.energy_per_atom * amt for c, amt in decomp.items()])
        decomp_enthalpy = entry.energy_per_atom - decomp_enthalpy
        return (decomp, decomp_enthalpy)

    def get_phase_separation_energy(self, entry, **kwargs):
        """
        Provides the energy to the convex hull for the given entry. For stable entries
        already in the phase diagram the algorithm provides the phase separation energy
        which is referred to as the decomposition enthalpy in:

        1. Bartel, C., Trewartha, A., Wang, Q., Dunn, A., Jain, A., Ceder, G.,
            A critical examination of compound stability predictions from
            machine-learned formation energies, npj Computational Materials 6, 97 (2020)

        Args:
            entry (PDEntry): A PDEntry like object
            **kwargs: Keyword args passed to `get_decomp_and_decomp_energy`
                space_limit (int): The maximum number of competing entries to consider.
                stable_only (bool): Only use stable materials as competing entries
                tol (float): The tolerance for convergence of the SLSQP optimization
                    when finding the equilibrium reaction.
                maxiter (int): The maximum number of iterations of the SLSQP optimizer
                    when finding the equilibrium reaction.

        Returns:
            phase separation energy per atom of entry. Stable entries should have
            energies <= 0, Stable elemental entries should have energies = 0 and
            unstable entries should have energies > 0. Entries that have the same
            composition as a stable energy may have positive or negative phase
            separation energies depending on their own energy.
        """
        return self.get_decomp_and_phase_separation_energy(entry, **kwargs)[1]

    def get_composition_chempots(self, comp):
        """
        Get the chemical potentials for all elements at a given composition.

        Args:
            comp (Composition): Composition

        Returns:
            Dictionary of chemical potentials.
        """
        facet = self._get_facet_and_simplex(comp)[0]
        return self._get_facet_chempots(facet)

    def get_all_chempots(self, comp):
        """
        Get chemical potentials at a given composition.

        Args:
            comp (Composition): Composition

        Returns:
            Chemical potentials.
        """
        all_facets = self._get_all_facets_and_simplexes(comp)
        chempots = {}
        for facet in all_facets:
            facet_name = '-'.join((self.qhull_entries[j].name for j in facet))
            chempots[facet_name] = self._get_facet_chempots(facet)
        return chempots

    def get_transition_chempots(self, element):
        """
        Get the critical chemical potentials for an element in the Phase
        Diagram.

        Args:
            element: An element. Has to be in the PD in the first place.

        Returns:
            A sorted sequence of critical chemical potentials, from less
            negative to more negative.
        """
        if element not in self.elements:
            raise ValueError('get_transition_chempots can only be called with elements in the phase diagram.')
        critical_chempots = []
        for facet in self.facets:
            chempots = self._get_facet_chempots(facet)
            critical_chempots.append(chempots[element])
        clean_pots = []
        for c in sorted(critical_chempots):
            if len(clean_pots) == 0 or abs(c - clean_pots[-1]) > PhaseDiagram.numerical_tol:
                clean_pots.append(c)
        clean_pots.reverse()
        return tuple(clean_pots)

    def get_critical_compositions(self, comp1, comp2):
        """
        Get the critical compositions along the tieline between two
        compositions. I.e. where the decomposition products change.
        The endpoints are also returned.

        Args:
            comp1 (Composition): First composition to define the tieline
            comp2 (Composition): Second composition to define the tieline

        Returns:
            [(Composition)]: list of critical compositions. All are of
                the form x * comp1 + (1-x) * comp2
        """
        n1 = comp1.num_atoms
        n2 = comp2.num_atoms
        pd_els = self.elements
        c1 = self.pd_coords(comp1)
        c2 = self.pd_coords(comp2)
        if np.all(c1 == c2):
            return [comp1.copy(), comp2.copy()]
        intersections = self._get_simplex_intersections(c1, c2)
        line = c2 - c1
        line /= np.sum(line ** 2) ** 0.5
        proj = np.dot(intersections - c1, line)
        proj = proj[np.logical_and(proj > -self.numerical_tol, proj < proj[1] + self.numerical_tol)]
        proj.sort()
        valid = np.ones(len(proj), dtype=bool)
        valid[1:] = proj[1:] > proj[:-1] + self.numerical_tol
        proj = proj[valid]
        ints = c1 + line * proj[:, None]
        cs = np.concatenate([np.array([1 - np.sum(ints, axis=-1)]).T, ints], axis=-1)
        x = proj / np.dot(c2 - c1, line)
        x_unnormalized = x * n1 / (n2 + x * (n1 - n2))
        num_atoms = n1 + (n2 - n1) * x_unnormalized
        cs *= num_atoms[:, None]
        return [Composition(((elem, val) for elem, val in zip(pd_els, m))) for m in cs]

    def get_element_profile(self, element, comp, comp_tol=1e-05):
        """
        Provides the element evolution data for a composition. For example, can be used
        to analyze Li conversion voltages by varying mu_Li and looking at the phases
        formed. Also can be used to analyze O2 evolution by varying mu_O2.

        Args:
            element: An element. Must be in the phase diagram.
            comp: A Composition
            comp_tol: The tolerance to use when calculating decompositions.
                Phases with amounts less than this tolerance are excluded.
                Defaults to 1e-5.

        Returns:
            Evolution data as a list of dictionaries of the following format:
            [ {'chempot': -10.487582, 'evolution': -2.0,
            'reaction': Reaction Object], ...]
        """
        element = get_el_sp(element)
        if element not in self.elements:
            raise ValueError('get_transition_chempots can only be called with elements in the phase diagram.')
        gc_comp = Composition({el: amt for el, amt in comp.items() if el != element})
        el_ref = self.el_refs[element]
        el_comp = Composition(element.symbol)
        evolution = []
        for cc in self.get_critical_compositions(el_comp, gc_comp)[1:]:
            decomp_entries = list(self.get_decomposition(cc))
            decomp = [k.composition for k in decomp_entries]
            rxn = Reaction([comp], [*decomp, el_comp])
            rxn.normalize_to(comp)
            c = self.get_composition_chempots(cc + el_comp * 1e-05)[element]
            amt = -rxn.coeffs[rxn.all_comp.index(el_comp)]
            evolution.append({'chempot': c, 'evolution': amt, 'element_reference': el_ref, 'reaction': rxn, 'entries': decomp_entries, 'critical_composition': cc})
        return evolution

    def get_chempot_range_map(self, elements: Sequence[Element], referenced: bool=True, joggle: bool=True) -> dict[Element, list[Simplex]]:
        """
        Returns a chemical potential range map for each stable entry.

        Args:
            elements: Sequence of elements to be considered as independent variables.
                E.g., if you want to show the stability ranges
                of all Li-Co-O phases with respect to mu_Li and mu_O, you will supply
                [Element("Li"), Element("O")]
            referenced: If True, gives the results with a reference being the
                energy of the elemental phase. If False, gives absolute values.
            joggle (bool): Whether to joggle the input to avoid precision
                errors.

        Returns:
            Returns a dict of the form {entry: [simplices]}. The list of
            simplices are the sides of the N-1 dim polytope bounding the
            allowable chemical potential range of each entry.
        """
        all_chempots = []
        for facet in self.facets:
            chempots = self._get_facet_chempots(facet)
            all_chempots.append([chempots[el] for el in self.elements])
        inds = [self.elements.index(el) for el in elements]
        if referenced:
            el_energies = {el: self.el_refs[el].energy_per_atom for el in elements}
        else:
            el_energies = dict.fromkeys(elements, 0)
        chempot_ranges = collections.defaultdict(list)
        vertices = [list(range(len(self.elements)))]
        if len(all_chempots) > len(self.elements):
            vertices = get_facets(all_chempots, joggle=joggle)
        for ufacet in vertices:
            for combi in itertools.combinations(ufacet, 2):
                data1 = self.facets[combi[0]]
                data2 = self.facets[combi[1]]
                common_ent_ind = set(data1).intersection(set(data2))
                if len(common_ent_ind) == len(elements):
                    common_entries = [self.qhull_entries[idx] for idx in common_ent_ind]
                    data = np.array([[all_chempots[ii][jj] - el_energies[self.elements[jj]] for jj in inds] for ii in combi])
                    sim = Simplex(data)
                    for entry in common_entries:
                        chempot_ranges[entry].append(sim)
        return chempot_ranges

    def getmu_vertices_stability_phase(self, target_comp, dep_elt, tol_en=0.01):
        """
        Returns a set of chemical potentials corresponding to the vertices of
        the simplex in the chemical potential phase diagram.
        The simplex is built using all elements in the target_composition
        except dep_elt.
        The chemical potential of dep_elt is computed from the target
        composition energy.
        This method is useful to get the limiting conditions for
        defects computations for instance.

        Args:
            target_comp: A Composition object
            dep_elt: the element for which the chemical potential is computed
                from the energy of the stable phase at the target composition
            tol_en: a tolerance on the energy to set

        Returns:
            [{Element: mu}]: An array of conditions on simplex vertices for
            which each element has a chemical potential set to a given
            value. "absolute" values (i.e., not referenced to element energies)
        """
        mu_ref = np.array([self.el_refs[elem].energy_per_atom for elem in self.elements if elem != dep_elt])
        chempot_ranges = self.get_chempot_range_map([elem for elem in self.elements if elem != dep_elt])
        for elem in self.elements:
            if elem not in target_comp.elements:
                target_comp = target_comp + Composition({elem: 0.0})
        coeff = [-target_comp[elem] for elem in self.elements if elem != dep_elt]
        for elem, chempots in chempot_ranges.items():
            if elem.composition.reduced_composition == target_comp.reduced_composition:
                multiplier = elem.composition[dep_elt] / target_comp[dep_elt]
                ef = elem.energy / multiplier
                all_coords = []
                for simplex in chempots:
                    for v in simplex._coords:
                        elements = [elem for elem in self.elements if elem != dep_elt]
                        res = {}
                        for idx, el in enumerate(elements):
                            res[el] = v[idx] + mu_ref[idx]
                        res[dep_elt] = (np.dot(v + mu_ref, coeff) + ef) / target_comp[dep_elt]
                        already_in = False
                        for di in all_coords:
                            dict_equals = True
                            for k in di:
                                if abs(di[k] - res[k]) > tol_en:
                                    dict_equals = False
                                    break
                            if dict_equals:
                                already_in = True
                                break
                        if not already_in:
                            all_coords.append(res)
        return all_coords

    def get_chempot_range_stability_phase(self, target_comp, open_elt):
        """
        Returns a set of chemical potentials corresponding to the max and min
        chemical potential of the open element for a given composition. It is
        quite common to have for instance a ternary oxide (e.g., ABO3) for
        which you want to know what are the A and B chemical potential leading
        to the highest and lowest oxygen chemical potential (reducing and
        oxidizing conditions). This is useful for defect computations.

        Args:
            target_comp: A Composition object
            open_elt: Element that you want to constrain to be max or min

        Returns:
            {Element: (mu_min, mu_max)}: Chemical potentials are given in
                "absolute" values (i.e., not referenced to 0)
        """
        mu_ref = np.array([self.el_refs[elem].energy_per_atom for elem in self.elements if elem != open_elt])
        chempot_ranges = self.get_chempot_range_map([elem for elem in self.elements if elem != open_elt])
        for elem in self.elements:
            if elem not in target_comp.elements:
                target_comp = target_comp + Composition({elem: 0.0})
        coeff = [-target_comp[elem] for elem in self.elements if elem != open_elt]
        max_open = -float('inf')
        min_open = float('inf')
        max_mus = min_mus = None
        for elem, chempots in chempot_ranges.items():
            if elem.composition.reduced_composition == target_comp.reduced_composition:
                multiplier = elem.composition[open_elt] / target_comp[open_elt]
                ef = elem.energy / multiplier
                all_coords = []
                for s in chempots:
                    for v in s._coords:
                        all_coords.append(v)
                        test_open = (np.dot(v + mu_ref, coeff) + ef) / target_comp[open_elt]
                        if test_open > max_open:
                            max_open = test_open
                            max_mus = v
                        if test_open < min_open:
                            min_open = test_open
                            min_mus = v
        elems = [elem for elem in self.elements if elem != open_elt]
        res = {}
        for idx, el in enumerate(elems):
            res[el] = (min_mus[idx] + mu_ref[idx], max_mus[idx] + mu_ref[idx])
        res[open_elt] = (min_open, max_open)
        return res

    def get_plot(self, show_unstable: float=0.2, backend: Literal['plotly', 'matplotlib']='plotly', ternary_style: Literal['2d', '3d']='2d', label_stable: bool=True, label_unstable: bool=True, ordering: Sequence[str] | None=None, energy_colormap=None, process_attributes: bool=False, ax: plt.Axes=None, label_uncertainties: bool=False, fill: bool=True, **kwargs):
        """
        Convenient wrapper for PDPlotter. Initializes a PDPlotter object and calls
        get_plot() with provided combined arguments.

        Plotting is only supported for phase diagrams with <=4 elements (unary,
        binary, ternary, or quaternary systems).

        Args:
            show_unstable (float): Whether unstable (above the hull) phases will be
                plotted. If a number > 0 is entered, all phases with
                e_hull < show_unstable (eV/atom) will be shown.
            backend ("plotly" | "matplotlib"): Python package to use for plotting.
                Defaults to "plotly".
            ternary_style ("2d" | "3d"): Ternary phase diagrams are typically plotted in
                two-dimensions (2d), but can be plotted in three dimensions (3d) to visualize
                the depth of the hull. This argument only applies when backend="plotly".
                Defaults to "2d".
            label_stable: Whether to label stable compounds.
            label_unstable: Whether to label unstable compounds.
            ordering: Ordering of vertices (matplotlib backend only).
            energy_colormap: Colormap for coloring energy (matplotlib backend only).
            process_attributes: Whether to process the attributes (matplotlib
                backend only).
            ax: Existing Axes object if plotting multiple phase diagrams (matplotlib backend only).
            label_uncertainties: Whether to add error bars to the hull (plotly
                backend only). For binaries, this also shades the hull with the
                uncertainty window.
            fill: Whether to shade the hull. For ternary_2d and quaternary plots, this
                colors facets arbitrarily for visual clarity. For ternary_3d plots, this
                shades the hull by formation energy (plotly backend only).
            **kwargs (dict): Keyword args passed to PDPlotter.get_plot(). Can be used to customize markers
                etc. If not set, the default is { "markerfacecolor": "#4daf4a", "markersize": 10, "linewidth": 3 }
        """
        plotter = PDPlotter(self, show_unstable=show_unstable, backend=backend, ternary_style=ternary_style)
        return plotter.get_plot(label_stable=label_stable, label_unstable=label_unstable, ordering=ordering, energy_colormap=energy_colormap, process_attributes=process_attributes, ax=ax, label_uncertainties=label_uncertainties, fill=fill, **kwargs)