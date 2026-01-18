from __future__ import annotations
import abc
import copy
import os
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Union
import numpy as np
from monty.design_patterns import cached_class
from monty.json import MSONable
from monty.serialization import loadfn
from tqdm import tqdm
from uncertainties import ufloat
from pymatgen.analysis.structure_analyzer import oxide_type, sulfide_type
from pymatgen.core import SETTINGS, Composition, Element
from pymatgen.entries.computed_entries import (
from pymatgen.io.vasp.sets import MITRelaxSet, MPRelaxSet, VaspInputSet
from pymatgen.util.due import Doi, due
@cached_class
@due.dcite(Doi('10.1103/PhysRevB.85.235438', 'Pourbaix scheme to combine calculated and experimental data'))
class MaterialsProjectAqueousCompatibility(Compatibility):
    """This class implements the Aqueous energy referencing scheme for constructing
    Pourbaix diagrams from DFT energies, as described in Persson et al.

    This scheme applies various energy adjustments to convert DFT energies into
    Gibbs free energies of formation at 298 K and to guarantee that the experimental
    formation free energy of H2O is reproduced. Briefly, the steps are:

        1. Beginning with the DFT energy of O2, adjust the energy of H2 so that
           the experimental reaction energy of -2.458 eV/H2O is reproduced.
        2. Add entropy to the DFT energy of any compounds that are liquid or
           gaseous at room temperature
        3. Adjust the DFT energies of solid hydrate compounds (compounds that
           contain water, e.g. FeO.nH2O) such that the energies of the embedded
           H2O molecules are equal to the experimental free energy

    The above energy adjustments are computed dynamically based on the input
    Entries.

    References:
        K.A. Persson, B. Waldwick, P. Lazic, G. Ceder, Prediction of solid-aqueous
        equilibria: Scheme to combine first-principles calculations of solids with
        experimental aqueous states, Phys. Rev. B - Condens. Matter Mater. Phys.
        85 (2012) 1-12. doi:10.1103/PhysRevB.85.235438.
    """

    def __init__(self, solid_compat: Compatibility | type[Compatibility] | None=MaterialsProject2020Compatibility, o2_energy: float | None=None, h2o_energy: float | None=None, h2o_adjustments: float | None=None) -> None:
        """Initialize the MaterialsProjectAqueousCompatibility class.

        Note that this class requires as inputs the ground-state DFT energies of O2 and H2O, plus the value of any
        energy adjustments applied to an H2O molecule. If these parameters are not provided in __init__, they can
        be automatically populated by including ComputedEntry for the ground state of O2 and H2O in a list of
        entries passed to process_entries. process_entries will fail if one or the other is not provided.

        Args:
            solid_compat: Compatibility scheme used to pre-process solid DFT energies prior to applying aqueous
                energy adjustments. May be passed as a class (e.g. MaterialsProject2020Compatibility) or an instance
                (e.g., MaterialsProject2020Compatibility()). If None, solid DFT energies are used as-is.
                Default: MaterialsProject2020Compatibility
            o2_energy: The ground-state DFT energy of oxygen gas, including any adjustments or corrections, in eV/atom.
                If not set, this value will be determined from any O2 entries passed to process_entries.
                Default: None
            h2o_energy: The ground-state DFT energy of water, including any adjustments or corrections, in eV/atom.
                If not set, this value will be determined from any H2O entries passed to process_entries.
                Default: None
            h2o_adjustments: Total energy adjustments applied to one water molecule, in eV/atom.
                If not set, this value will be determined from any H2O entries passed to process_entries.
                Default: None
        """
        self.solid_compat = None
        if solid_compat is None:
            self.solid_compat = None
        elif isinstance(solid_compat, type) and issubclass(solid_compat, Compatibility):
            self.solid_compat = solid_compat()
        elif issubclass(type(solid_compat), Compatibility):
            self.solid_compat = solid_compat
        else:
            raise ValueError('Expected a Compatibility class, instance of a Compatibility or None')
        self.o2_energy = o2_energy
        self.h2o_energy = h2o_energy
        self.h2_energy = None
        self.h2o_adjustments = h2o_adjustments
        if not all([self.o2_energy, self.h2o_energy, self.h2o_adjustments]):
            warnings.warn(f'You did not provide the required O2 and H2O energies. {type(self).__name__} needs these energies in order to compute the appropriate energy adjustments. It will try to determine the values from ComputedEntry for O2 and H2O passed to process_entries, but will fail if these entries are not provided.')
        self.cpd_entropies = {'O2': 0.316731, 'N2': 0.295729, 'F2': 0.313025, 'Cl2': 0.344373, 'Br': 0.235039, 'Hg': 0.234421, 'H2O': 0.071963}
        self.name = 'MP Aqueous free energy adjustment'
        super().__init__()

    def get_adjustments(self, entry: ComputedEntry) -> list[EnergyAdjustment]:
        """Returns the corrections applied to a particular entry.

        Args:
            entry: A ComputedEntry object.

        Returns:
            list[EnergyAdjustment]: Energy adjustments to be applied to entry.

        Raises:
            CompatibilityError if the required O2 and H2O energies have not been provided to
            MaterialsProjectAqueousCompatibility during init or in the list of entries passed to process_entries.
        """
        adjustments = []
        if self.o2_energy is None or self.h2o_energy is None or self.h2o_adjustments is None:
            raise CompatibilityError(f'You did not provide the required O2 and H2O energies. {type(self).__name__} needs these energies in order to compute the appropriate energy adjustments. Either specify the energies as arguments to {type(self).__name__}.__init__ or run process_entries on a list that includes ComputedEntry for the ground state of O2 and H2O.')
        self.fit_h2_energy = round(0.5 * (3 * (self.h2o_energy - self.cpd_entropies['H2O']) - (self.o2_energy - self.cpd_entropies['O2']) - MU_H2O), 6)
        comp = entry.composition
        rform = comp.reduced_formula
        if rform == 'H2':
            assert self.h2_energy is not None, 'H2 energy not set'
            adjustments.append(ConstantEnergyAdjustment((self.fit_h2_energy - self.h2_energy) * comp.num_atoms, uncertainty=np.nan, name='MP Aqueous H2 / H2O referencing', cls=self.as_dict(), description='Adjusts the H2 energy to reproduce the experimental Gibbs formation free energy of H2O, based on the DFT energy of Oxygen and H2O'))
        if rform in self.cpd_entropies:
            adjustments.append(TemperatureEnergyAdjustment(-self.cpd_entropies[rform] / 298, 298, comp.num_atoms, uncertainty_per_deg=np.nan, name='Compound entropy at room temperature', cls=self.as_dict(), description='Adds the entropy (T delta S) to energies of compounds that are gaseous or liquid at standard state'))
        if rform != 'H2O':
            rcomp, factor = comp.get_reduced_composition_and_factor()
            nH2O = int(min(rcomp['H'] / 2.0, rcomp['O'])) * factor
            if nH2O > 0:
                hydrate_adjustment = -1 * (self.h2o_adjustments * 3 + MU_H2O)
                adjustments.append(CompositionEnergyAdjustment(hydrate_adjustment, nH2O, uncertainty_per_atom=np.nan, name='MP Aqueous hydrate', cls=self.as_dict(), description='Adjust the energy of solid hydrate compounds (compounds containing H2O molecules in their structure) so that the free energies of embedded H2O molecules match the experimental value enforced by the MP Aqueous energy referencing scheme.'))
        return adjustments

    def process_entries(self, entries: list[AnyComputedEntry], clean: bool=False, verbose: bool=False, inplace: bool=True, on_error: Literal['ignore', 'warn', 'raise']='ignore') -> list[AnyComputedEntry]:
        """Process a sequence of entries with the chosen Compatibility scheme.

        Args:
            entries (list[ComputedEntry | ComputedStructureEntry]): Entries to be processed.
            clean (bool): Whether to remove any previously-applied energy adjustments.
                If True, all EnergyAdjustment are removed prior to processing the Entry.
                Default is False.
            verbose (bool): Whether to display progress bar for processing multiple entries.
                Default is False.
            inplace (bool): Whether to modify the entries in place. If False, a copy of the
                entries is made and processed. Default is True.
            on_error ('ignore' | 'warn' | 'raise'): What to do when get_adjustments(entry)
                raises CompatibilityError. Defaults to 'ignore'.

        Returns:
            list[AnyComputedEntry]: Adjusted entries. Entries in the original list incompatible with
                chosen correction scheme are excluded from the returned list.
        """
        if isinstance(entries, ComputedEntry):
            entries = [entries]
        if not inplace:
            entries = copy.deepcopy(entries)
        if self.solid_compat:
            entries = self.solid_compat.process_entries(entries, clean=True)
        if len(entries) == 1 and entries[0].reduced_formula == 'H2':
            warnings.warn('Processing single H2 entries will result in the all polymorphs being assigned the same energy. This should not cause problems with Pourbaix diagram construction, but may be confusing. Pass all entries to process_entries() at once in if you want to preserve H2 polymorph energy differences.')
        if len(entries) > 1:
            if not self.o2_energy:
                o2_entries = [e for e in entries if e.reduced_formula == 'O2']
                if o2_entries:
                    self.o2_energy = min((e.energy_per_atom for e in o2_entries))
            if not self.h2o_energy and (not self.h2o_adjustments):
                h2o_entries = [e for e in entries if e.reduced_formula == 'H2O']
                if h2o_entries:
                    h2o_entries = sorted(h2o_entries, key=lambda e: e.energy_per_atom)
                    self.h2o_energy = h2o_entries[0].energy_per_atom
                    self.h2o_adjustments = h2o_entries[0].correction / h2o_entries[0].composition.num_atoms
        h2_entries = [e for e in entries if e.reduced_formula == 'H2']
        if h2_entries:
            h2_entries = sorted(h2_entries, key=lambda e: e.energy_per_atom)
            self.h2_energy = h2_entries[0].energy_per_atom
        return super().process_entries(entries, clean=clean, verbose=verbose, inplace=inplace, on_error=on_error)