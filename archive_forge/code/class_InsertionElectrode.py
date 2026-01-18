from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING
from monty.json import MontyDecoder
from scipy.constants import N_A
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.apps.battery.battery_abc import AbstractElectrode, AbstractVoltagePair
from pymatgen.core import Composition, Element
from pymatgen.core.units import Charge, Time
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
@dataclass
class InsertionElectrode(AbstractElectrode):
    """A set of topotactically related compounds, with different amounts of a
    single element, e.g. TiO2 and LiTiO2, that can be used to define an
    insertion battery electrode.
    """
    stable_entries: Iterable[ComputedEntry]
    unstable_entries: Iterable[ComputedEntry]

    @classmethod
    def from_entries(cls, entries: Iterable[ComputedEntry | ComputedStructureEntry], working_ion_entry: ComputedEntry | ComputedStructureEntry | PDEntry, strip_structures: bool=False) -> Self:
        """Create a new InsertionElectrode.

        Args:
            entries: A list of ComputedEntries, ComputedStructureEntries, or
                subclasses representing the different topotactic states
                of the battery, e.g. TiO2 and LiTiO2.
            working_ion_entry: A single ComputedEntry or PDEntry
                representing the element that carries charge across the
                battery, e.g. Li.
            strip_structures: Since the electrode document only uses volume we can make the
                electrode object significantly leaner by dropping the structure data.
                If this parameter is set to True, the ComputedStructureEntry will be
                replaced with a ComputedEntry and the volume will be stored in
                ComputedEntry.data['volume']. If entries provided are ComputedEntries,
                must set strip_structures=False.
        """
        if strip_structures:
            ents = []
            for ient in entries:
                dd = ient.as_dict()
                ent = ComputedEntry.from_dict(dd)
                ent.data['volume'] = ient.structure.volume
                ents.append(ent)
            entries = ents
        _working_ion = working_ion_entry.elements[0]
        _working_ion_entry = working_ion_entry
        elements = set()
        for entry in entries:
            elements.update(entry.elements)
        element_energy = max((entry.energy_per_atom for entry in entries)) + 10
        pdentries: list[ComputedEntry | ComputedStructureEntry | PDEntry] = []
        pdentries.extend(entries)
        pdentries.extend([PDEntry(Composition({el: 1}), element_energy) for el in elements])
        pd = PhaseDiagram(pdentries)

        def lifrac(e):
            return e.composition.get_atomic_fraction(_working_ion)
        _stable_entries = tuple(sorted((e for e in pd.stable_entries if e in entries), key=lifrac))
        _unstable_entries = tuple(sorted((e for e in pd.unstable_entries if e in entries), key=lifrac))
        _vpairs: tuple[AbstractVoltagePair, ...] = tuple((InsertionVoltagePair.from_entries(_stable_entries[i], _stable_entries[i + 1], working_ion_entry) for i in range(len(_stable_entries) - 1)))
        framework = _vpairs[0].framework
        return cls(voltage_pairs=_vpairs, working_ion_entry=_working_ion_entry, stable_entries=_stable_entries, unstable_entries=_unstable_entries, framework_formula=framework.reduced_formula)

    def get_stable_entries(self, charge_to_discharge=True):
        """Get the stable entries.

        Args:
            charge_to_discharge: order from most charge to most discharged
                state? Default to True.

        Returns:
            A list of stable entries in the electrode, ordered by amount of the
            working ion.
        """
        list_copy = list(self.stable_entries)
        return list_copy if charge_to_discharge else list_copy.reverse()

    def get_unstable_entries(self, charge_to_discharge=True):
        """Returns the unstable entries for the electrode.

        Args:
            charge_to_discharge: Order from most charge to most discharged
                state? Defaults to True.

        Returns:
            A list of unstable entries in the electrode, ordered by amount of
            the working ion.
        """
        list_copy = list(self.unstable_entries)
        return list_copy if charge_to_discharge else list_copy.reverse()

    def get_all_entries(self, charge_to_discharge=True):
        """Return all entries input for the electrode.

        Args:
            charge_to_discharge:
                order from most charge to most discharged state? Defaults to
                True.

        Returns:
            A list of all entries in the electrode (both stable and unstable),
            ordered by amount of the working ion.
        """
        all_entries = list(self.get_stable_entries())
        all_entries.extend(self.get_unstable_entries())
        all_entries = sorted(all_entries, key=lambda e: e.composition.get_atomic_fraction(self.working_ion))
        return all_entries if charge_to_discharge else all_entries.reverse()

    @property
    def fully_charged_entry(self):
        """The most charged entry along the topotactic path."""
        return self.stable_entries[0]

    @property
    def fully_discharged_entry(self):
        """The most discharged entry along the topotactic path."""
        return self.stable_entries[-1]

    def get_max_instability(self, min_voltage=None, max_voltage=None):
        """The maximum instability along a path for a specific voltage range.

        Args:
            min_voltage: The minimum allowable voltage.
            max_voltage: The maximum allowable voltage.

        Returns:
            Maximum decomposition energy of all compounds along the insertion
            path (a subset of the path can be chosen by the optional arguments)
        """
        data = []
        for pair in self._select_in_voltage_range(min_voltage, max_voltage):
            if getattr(pair, 'decomp_e_charge', None) is not None:
                data.append(pair.decomp_e_charge)
            if getattr(pair, 'decomp_e_discharge', None) is not None:
                data.append(pair.decomp_e_discharge)
        return max(data) if len(data) > 0 else None

    def get_min_instability(self, min_voltage=None, max_voltage=None):
        """The minimum instability along a path for a specific voltage range.

        Args:
            min_voltage: The minimum allowable voltage.
            max_voltage: The maximum allowable voltage.

        Returns:
            Minimum decomposition energy of all compounds along the insertion
            path (a subset of the path can be chosen by the optional arguments)
        """
        data = []
        for pair in self._select_in_voltage_range(min_voltage, max_voltage):
            if getattr(pair, 'decomp_e_charge', None) is not None:
                data.append(pair.decomp_e_charge)
            if getattr(pair, 'decomp_e_discharge', None) is not None:
                data.append(pair.decomp_e_discharge)
        return min(data) if len(data) > 0 else None

    def get_max_muO2(self, min_voltage=None, max_voltage=None):
        """Maximum critical oxygen chemical potential along path.

        Args:
            min_voltage: The minimum allowable voltage.
            max_voltage: The maximum allowable voltage.

        Returns:
            Maximum critical oxygen chemical of all compounds along the
            insertion path (a subset of the path can be chosen by the optional
            arguments).
        """
        data = []
        for pair in self._select_in_voltage_range(min_voltage, max_voltage):
            if pair.muO2_discharge is not None:
                data.extend([d['chempot'] for d in pair.muO2_discharge])
            if pair.muO2_charge is not None:
                data.extend([d['chempot'] for d in pair.muO2_discharge])
        return max(data) if len(data) > 0 else None

    def get_min_muO2(self, min_voltage=None, max_voltage=None):
        """Minimum critical oxygen chemical potential along path.

        Args:
            min_voltage: The minimum allowable voltage for a given step
            max_voltage: The maximum allowable voltage allowable for a given
                step

        Returns:
            Minimum critical oxygen chemical of all compounds along the
            insertion path (a subset of the path can be chosen by the optional
            arguments).
        """
        data = []
        for pair in self._select_in_voltage_range(min_voltage, max_voltage):
            if pair.muO2_discharge is not None:
                data.extend([d['chempot'] for d in pair.muO2_discharge])
            if pair.muO2_charge is not None:
                data.extend([d['chempot'] for d in pair.muO2_discharge])
        return min(data) if len(data) > 0 else None

    def get_sub_electrodes(self, adjacent_only=True, include_myself=True):
        """If this electrode contains multiple voltage steps, then it is possible
        to use only a subset of the voltage steps to define other electrodes.
        For example, an LiTiO2 electrode might contain three subelectrodes:
        [LiTiO2 --> TiO2, LiTiO2 --> Li0.5TiO2, Li0.5TiO2 --> TiO2]
        This method can be used to return all the subelectrodes with some
        options.

        Args:
            adjacent_only: Only return electrodes from compounds that are
                adjacent on the convex hull, i.e. no electrodes returned
                will have multiple voltage steps if this is set True.
            include_myself: Include this identical electrode in the list of
                results.

        Returns:
            A list of InsertionElectrode objects
        """
        battery_list = []
        pair_it = self.voltage_pairs if adjacent_only else itertools.combinations_with_replacement(self.voltage_pairs, 2)
        ion = self.working_ion
        for pair in pair_it:
            entry_charge = pair.entry_charge if adjacent_only else pair[0].entry_charge
            entry_discharge = pair.entry_discharge if adjacent_only else pair[1].entry_discharge

            def in_range(entry):
                chg_frac = entry_charge.composition.get_atomic_fraction(ion)
                dischg_frac = entry_discharge.composition.get_atomic_fraction(ion)
                frac = entry.composition.get_atomic_fraction(ion)
                return chg_frac <= frac <= dischg_frac
            if include_myself or entry_charge != self.fully_charged_entry or entry_discharge != self.fully_discharged_entry:
                unstable_entries = filter(in_range, self.get_unstable_entries())
                stable_entries = filter(in_range, self.get_stable_entries())
                all_entries = list(stable_entries)
                all_entries.extend(unstable_entries)
                battery_list.append(type(self).from_entries(all_entries, self.working_ion_entry))
        return battery_list

    def get_summary_dict(self, print_subelectrodes=True) -> dict:
        """Generate a summary dict.
        Populates the summary dict with the basic information from the parent method then populates more information.
        Since the parent method calls self.get_summary_dict(print_subelectrodes=True) for the subelectrodes.
        The current method will be called from within super().get_summary_dict.

        Args:
            print_subelectrodes: Also print data on all the possible
                subelectrodes.

        Returns:
            A summary of this electrode's properties in dict format.
        """
        dct = super().get_summary_dict(print_subelectrodes=print_subelectrodes)
        chg_comp = self.fully_charged_entry.composition
        dischg_comp = self.fully_discharged_entry.composition
        dct.update({'id_charge': self.fully_charged_entry.entry_id, 'formula_charge': chg_comp.reduced_formula, 'id_discharge': self.fully_discharged_entry.entry_id, 'formula_discharge': dischg_comp.reduced_formula, 'max_instability': self.get_max_instability(), 'min_instability': self.get_min_instability(), 'material_ids': [itr_ent.entry_id for itr_ent in self.get_all_entries()], 'stable_material_ids': [itr_ent.entry_id for itr_ent in self.get_stable_entries()], 'unstable_material_ids': [itr_ent.entry_id for itr_ent in self.get_unstable_entries()]})
        if all(('decomposition_energy' in itr_ent.data for itr_ent in self.get_all_entries())):
            dct.update(stability_charge=self.fully_charged_entry.data['decomposition_energy'], stability_discharge=self.fully_discharged_entry.data['decomposition_energy'], stability_data={itr_ent.entry_id: itr_ent.data['decomposition_energy'] for itr_ent in self.get_all_entries()})
        if all(('muO2' in itr_ent.data for itr_ent in self.get_all_entries())):
            dct.update({'muO2_data': {itr_ent.entry_id: itr_ent.data['muO2'] for itr_ent in self.get_all_entries()}})
        return dct

    def __repr__(self):
        chg_form = self.fully_charged_entry.reduced_formula
        dischg_form = self.fully_discharged_entry.reduced_formula
        return f'InsertionElectrode with endpoints at {chg_form} and {dischg_form}\nAvg. volt. = {self.get_average_voltage()} V\nGrav. cap. = {self.get_capacity_grav()} mAh/g\nVol. cap. = {self.get_capacity_vol()}'

    @classmethod
    def from_dict_legacy(cls, dct) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            InsertionElectrode
        """
        return InsertionElectrode(MontyDecoder().process_decoded(dct['entries']), MontyDecoder().process_decoded(dct['working_ion_entry']))

    def as_dict_legacy(self):
        """Returns: MSONable dict."""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'entries': [entry.as_dict() for entry in self.get_all_entries()], 'working_ion_entry': self.working_ion_entry.as_dict()}