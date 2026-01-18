from __future__ import annotations
import itertools
import logging
import re
import warnings
from copy import deepcopy
from functools import cmp_to_key, partial
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.special import comb
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import Composition, Element
from pymatgen.core.ion import Ion
from pymatgen.entries.compatibility import MU_H2O
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.util.coord import Simplex
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import Stringify
class PourbaixEntry(MSONable, Stringify):
    """
    An object encompassing all data relevant to a solid or ion
    in a Pourbaix diagram. Each bulk solid/ion has an energy
    g of the form: e = e0 + 0.0591 log10(conc) - nO mu_H2O
    + (nH - 2nO) pH + phi (-nH + 2nO + q).

    Note that the energies corresponding to the input entries
    should be formation energies with respect to hydrogen and
    oxygen gas in order for the Pourbaix diagram formalism to
    work. This may be changed to be more flexible in the future.
    """

    def __init__(self, entry, entry_id=None, concentration=1e-06):
        """
        Args:
            entry (ComputedEntry/ComputedStructureEntry/PDEntry/IonEntry): An
                entry object
            entry_id ():
            concentration ():
        """
        self.entry = entry
        if isinstance(entry, IonEntry):
            self.concentration = concentration
            self.phase_type = 'Ion'
            self.charge = entry.ion.charge
        else:
            self.concentration = 1.0
            self.phase_type = 'Solid'
            self.charge = 0
        self.uncorrected_energy = entry.energy
        if entry_id is not None:
            self.entry_id = entry_id
        elif hasattr(entry, 'entry_id') and entry.entry_id:
            self.entry_id = entry.entry_id
        else:
            self.entry_id = None

    @property
    def npH(self):
        """Get the number of H."""
        return self.entry.composition.get('H', 0) - 2 * self.entry.composition.get('O', 0)

    @property
    def nH2O(self):
        """Get the number of H2O."""
        return self.entry.composition.get('O', 0)

    @property
    def nPhi(self):
        """Get the number of electrons."""
        return self.npH - self.charge

    @property
    def name(self):
        """Get the name for entry."""
        if self.phase_type == 'Solid':
            return f'{self.entry.reduced_formula}(s)'
        return self.entry.name

    @property
    def energy(self):
        """Total energy of the Pourbaix entry (at pH, V = 0 vs. SHE)."""
        return self.uncorrected_energy + self.conc_term - MU_H2O * self.nH2O

    @property
    def energy_per_atom(self):
        """Energy per atom of the Pourbaix entry."""
        return self.energy / self.composition.num_atoms

    @property
    def elements(self):
        """Returns elements in the entry."""
        return self.entry.elements

    def energy_at_conditions(self, pH, V):
        """
        Get free energy for a given pH and V.

        Args:
            pH (float): pH at which to evaluate free energy
            V (float): voltage at which to evaluate free energy

        Returns:
            free energy at conditions
        """
        return self.energy + self.npH * PREFAC * pH + self.nPhi * V

    def get_element_fraction(self, element):
        """
        Gets the elemental fraction of a given non-OH element.

        Args:
            element (Element or str): string or element corresponding
                to element to get from composition

        Returns:
            fraction of element / sum(all non-OH elements)
        """
        return self.composition.get(element) * self.normalization_factor

    @property
    def normalized_energy(self):
        """
        Returns:
            energy normalized by number of non H or O atoms, e. g.
            for Zn2O6, energy / 2 or for AgTe3(OH)3, energy / 4.
        """
        return self.energy * self.normalization_factor

    def normalized_energy_at_conditions(self, pH, V):
        """
        Energy at an electrochemical condition, compatible with
        numpy arrays for pH/V input.

        Args:
            pH (float): pH at condition
            V (float): applied potential at condition

        Returns:
            energy normalized by number of non-O/H atoms at condition
        """
        return self.energy_at_conditions(pH, V) * self.normalization_factor

    @property
    def conc_term(self):
        """
        Returns the concentration contribution to the free energy,
        and should only be present when there are ions in the entry.
        """
        return PREFAC * np.log10(self.concentration)

    def as_dict(self):
        """
        Returns dict which contains Pourbaix Entry data.
        Note that the pH, voltage, H2O factors are always calculated when
        constructing a PourbaixEntry object.
        """
        dct = {'@module': type(self).__module__, '@class': type(self).__name__}
        if isinstance(self.entry, IonEntry):
            dct['entry_type'] = 'Ion'
        else:
            dct['entry_type'] = 'Solid'
        dct['entry'] = self.entry.as_dict()
        dct['concentration'] = self.concentration
        dct['entry_id'] = self.entry_id
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Invokes a PourbaixEntry from a dictionary."""
        entry_type = dct['entry_type']
        entry = IonEntry.from_dict(dct['entry']) if entry_type == 'Ion' else MontyDecoder().process_decoded(dct['entry'])
        entry_id = dct['entry_id']
        concentration = dct['concentration']
        return cls(entry, entry_id, concentration)

    @property
    def normalization_factor(self):
        """Sum of number of atoms minus the number of H and O in composition."""
        return 1.0 / (self.num_atoms - self.composition.get('H', 0) - self.composition.get('O', 0))

    @property
    def composition(self):
        """Returns composition."""
        return self.entry.composition

    @property
    def num_atoms(self):
        """Return number of atoms in current formula. Useful for normalization."""
        return self.composition.num_atoms

    def to_pretty_string(self) -> str:
        """A pretty string representation."""
        if self.phase_type == 'Solid':
            return f'{self.entry.reduced_formula}(s)'
        return self.entry.name

    def __repr__(self):
        energy, npH, nPhi, nH2O, entry_id = (self.energy, self.npH, self.nPhi, self.nH2O, self.entry_id)
        return f'{type(self).__name__}({self.entry.composition} with energy={energy:.4f}, npH={npH!r}, nPhi={nPhi!r}, nH2O={nH2O!r}, entry_id={entry_id!r})'