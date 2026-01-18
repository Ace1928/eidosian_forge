from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from scipy.constants import N_A
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.reaction_calculator import BalancedReaction
from pymatgen.apps.battery.battery_abc import AbstractElectrode, AbstractVoltagePair
from pymatgen.core import Composition, Element
from pymatgen.core.units import Charge, Time
@dataclass
class ConversionVoltagePair(AbstractVoltagePair):
    """A VoltagePair representing a Conversion Reaction with a defined voltage.
    Typically not initialized directly but rather used by ConversionElectrode.

    Attributes:
        rxn (BalancedReaction): BalancedReaction for the step
        voltage (float): Voltage for the step
        mAh (float): Capacity of the step
        vol_charge (float): Volume of charged state
        vol_discharge (float): Volume of discharged state
        mass_charge (float): Mass of charged state
        mass_discharge (float): Mass of discharged state
        frac_charge (float): Fraction of working ion in the charged state
        frac_discharge (float): Fraction of working ion in the discharged state
        entries_charge ([ComputedEntry]): Entries representing decompositions products
            in the charged state. Enumerates the decompositions products at the tieline,
            so the number of entries will be one fewer than the dimensions of the phase
            diagram
        entries_discharge ([ComputedEntry]): Entries representing decompositions products
            in the discharged state. Enumerates the decompositions products at the tieline,
            so the number of entries will be one fewer than the dimensions of the phase
            diagram
        working_ion_entry (ComputedEntry): Entry of the working ion.
    """
    rxn: BalancedReaction
    entries_charge: Iterable[ComputedEntry]
    entries_discharge: Iterable[ComputedEntry]

    @classmethod
    def from_steps(cls, step1, step2, normalization_els, framework_formula) -> Self:
        """Creates a ConversionVoltagePair from two steps in the element profile
        from a PD analysis.

        Args:
            step1: Starting step
            step2: Ending step
            normalization_els: Elements to normalize the reaction by. To
                ensure correct capacities.
            framework_formula: Formula of the framework.
        """
        working_ion_entry = step1['element_reference']
        working_ion = working_ion_entry.elements[0].symbol
        working_ion_valence = max(Element(working_ion).oxidation_states)
        voltage = (-step1['chempot'] + working_ion_entry.energy_per_atom) / working_ion_valence
        mAh = (step2['evolution'] - step1['evolution']) * Charge(1, 'e').to('C') * Time(1, 's').to('h') * N_A * 1000 * working_ion_valence
        li_comp = Composition(working_ion)
        prev_rxn = step1['reaction']
        reactants = {comp: abs(prev_rxn.get_coeff(comp)) for comp in prev_rxn.products if comp != li_comp}
        curr_rxn = step2['reaction']
        products = {comp: abs(curr_rxn.get_coeff(comp)) for comp in curr_rxn.products if comp != li_comp}
        reactants[li_comp] = step2['evolution'] - step1['evolution']
        rxn = BalancedReaction(reactants, products)
        for el, amt in normalization_els.items():
            if rxn.get_el_amount(el) > 1e-06:
                rxn.normalize_to_element(el, amt)
                break
        prev_mass_dischg = sum((prev_rxn.all_comp[idx].weight * abs(prev_rxn.coeffs[idx]) for idx in range(len(prev_rxn.all_comp)))) / 2
        vol_charge = sum((abs(prev_rxn.get_coeff(e.composition)) * e.structure.volume for e in step1['entries'] if e.reduced_formula != working_ion))
        mass_discharge = sum((curr_rxn.all_comp[idx].weight * abs(curr_rxn.coeffs[idx]) for idx in range(len(curr_rxn.all_comp)))) / 2
        mass_charge = prev_mass_dischg
        vol_discharge = sum((abs(curr_rxn.get_coeff(entry.composition)) * entry.structure.volume for entry in step2['entries'] if entry.reduced_formula != working_ion))
        total_comp = Composition()
        for comp in prev_rxn.products:
            if comp.reduced_formula != working_ion:
                total_comp += comp * abs(prev_rxn.get_coeff(comp))
        frac_charge = total_comp.get_atomic_fraction(Element(working_ion))
        total_comp = Composition()
        for comp in curr_rxn.products:
            if comp.reduced_formula != working_ion:
                total_comp += comp * abs(curr_rxn.get_coeff(comp))
        frac_discharge = total_comp.get_atomic_fraction(Element(working_ion))
        entries_charge = step1['entries']
        entries_discharge = step2['entries']
        return cls(rxn=rxn, voltage=voltage, mAh=mAh, vol_charge=vol_charge, vol_discharge=vol_discharge, mass_charge=mass_charge, mass_discharge=mass_discharge, frac_charge=frac_charge, frac_discharge=frac_discharge, entries_charge=entries_charge, entries_discharge=entries_discharge, working_ion_entry=working_ion_entry, framework_formula=framework_formula)

    def __repr__(self):
        output = [f'Conversion voltage pair with working ion {self.working_ion_entry.reduced_formula}', f'Reaction : {self.rxn}', f'V = {self.voltage}, mAh = {self.mAh}', f'frac_charge = {self.frac_charge}, frac_discharge = {self.frac_discharge}', f'mass_charge = {self.mass_charge}, mass_discharge = {self.mass_discharge}', f'vol_charge = {self.vol_charge}, vol_discharge = {self.vol_discharge}']
        return '\n'.join(output)