from __future__ import annotations
import re
import sys
import warnings
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from scipy.interpolate import InterpolatedUnivariateSpline
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.io.lmto import LMTOCopl
from pymatgen.io.lobster import Cohpcar
from pymatgen.util.coord import get_linear_interpolated_value
from pymatgen.util.due import Doi, due
from pymatgen.util.num import round_to_sigfigs
class CompleteCohp(Cohp):
    """A wrapper class that defines an average COHP, and individual COHPs.

    Attributes:
        are_coops (bool): Indicates whether the object is consisting of COOPs.
        are_cobis (bool): Indicates whether the object is consisting of COBIs.
        efermi (float): Fermi energy.
        energies (Sequence[float]): Sequence of energies.
        structure (pymatgen.Structure): Structure associated with the COHPs.
        cohp (Sequence[float]): The average COHP.
        icohp (Sequence[float]): The average ICOHP.
        all_cohps (dict[str, Sequence[float]]): A dict of COHPs for individual bonds of the form {label: COHP}.
        orb_res_cohp (dict[str, Dict[str, Sequence[float]]]): Orbital-resolved COHPs.
    """

    def __init__(self, structure, avg_cohp, cohp_dict, bonds=None, are_coops=False, are_cobis=False, are_multi_center_cobis=False, orb_res_cohp=None) -> None:
        """
        Args:
            structure: Structure associated with this COHP.
            avg_cohp: The average cohp as a COHP object.
            cohp_dict: A dict of COHP objects for individual bonds of the form
                {label: COHP}
            bonds: A dict containing information on the bonds of the form
                {label: {key: val}}. The key-val pair can be any information
                the user wants to put in, but typically contains the sites,
                the bond length, and the number of bonds. If nothing is
                supplied, it will default to an empty dict.
            are_coops: indicates whether the Cohp objects are COOPs.
                Defaults to False for COHPs.
            are_cobis: indicates whether the Cohp objects are COBIs.
                Defaults to False for COHPs.
            are_multi_center_cobis: indicates whether the Cohp objects are multi-center COBIs.
                Defaults to False for COHPs.
            orb_res_cohp: Orbital-resolved COHPs.
        """
        if are_coops and are_cobis or (are_coops and are_multi_center_cobis) or (are_cobis and are_multi_center_cobis):
            raise ValueError('You cannot have info about COOPs, COBIs and/or multi-center COBIS in the same file.')
        super().__init__(avg_cohp.efermi, avg_cohp.energies, avg_cohp.cohp, are_coops=are_coops, are_cobis=are_cobis, are_multi_center_cobis=are_multi_center_cobis, icohp=avg_cohp.icohp)
        self.structure = structure
        self.are_coops = are_coops
        self.are_cobis = are_cobis
        self.are_multi_center_cobis = are_multi_center_cobis
        self.all_cohps = cohp_dict
        self.orb_res_cohp = orb_res_cohp
        self.bonds = bonds or {label: {} for label in self.all_cohps}

    def __str__(self) -> str:
        if self.are_coops:
            return f'Complete COOPs for {self.structure}'
        if self.are_cobis:
            return f'Complete COBIs for {self.structure}'
        return f'Complete COHPs for {self.structure}'

    def as_dict(self):
        """JSON-serializable dict representation of CompleteCohp."""
        dct = {'@module': type(self).__module__, '@class': type(self).__name__, 'are_coops': self.are_coops, 'are_cobis': self.are_cobis, 'are_multi_center_cobis': self.are_multi_center_cobis, 'efermi': self.efermi, 'structure': self.structure.as_dict(), 'energies': self.energies.tolist(), 'COHP': {'average': {str(spin): pops.tolist() for spin, pops in self.cohp.items()}}}
        if self.icohp is not None:
            dct['ICOHP'] = {'average': {str(spin): pops.tolist() for spin, pops in self.icohp.items()}}
        for label in self.all_cohps:
            dct['COHP'].update({label: {str(spin): pops.tolist() for spin, pops in self.all_cohps[label].cohp.items()}})
            if self.all_cohps[label].icohp is not None:
                if 'ICOHP' not in dct:
                    dct['ICOHP'] = {label: {str(spin): pops.tolist() for spin, pops in self.all_cohps[label].icohp.items()}}
                else:
                    dct['ICOHP'].update({label: {str(spin): pops.tolist() for spin, pops in self.all_cohps[label].icohp.items()}})
        if False in [bond_dict == {} for bond_dict in self.bonds.values()]:
            dct['bonds'] = {bond: {'length': self.bonds[bond]['length'], 'sites': [site.as_dict() for site in self.bonds[bond]['sites']]} for bond in self.bonds}
        if self.orb_res_cohp:
            orb_dict = {}
            for label in self.orb_res_cohp:
                orb_dict[label] = {}
                for orbs in self.orb_res_cohp[label]:
                    cohp = {str(spin): pops.tolist() for spin, pops in self.orb_res_cohp[label][orbs]['COHP'].items()}
                    orb_dict[label][orbs] = {'COHP': cohp}
                    icohp = {str(spin): pops.tolist() for spin, pops in self.orb_res_cohp[label][orbs]['ICOHP'].items()}
                    orb_dict[label][orbs]['ICOHP'] = icohp
                    orbitals = [[orb[0], orb[1].name] for orb in self.orb_res_cohp[label][orbs]['orbitals']]
                    orb_dict[label][orbs]['orbitals'] = orbitals
            dct['orb_res_cohp'] = orb_dict
        return dct

    def get_cohp_by_label(self, label, summed_spin_channels=False):
        """Get specific COHP object.

        Args:
            label: string (for newer Lobster versions: a number)
            summed_spin_channels: bool, will sum the spin channels and return the sum in Spin.up if true

        Returns:
            Returns the COHP object to simplify plotting
        """
        if label.lower() == 'average':
            divided_cohp = self.cohp
            divided_icohp = self.icohp
        else:
            divided_cohp = self.all_cohps[label].get_cohp(spin=None, integrated=False)
            divided_icohp = self.all_cohps[label].get_icohp(spin=None)
        if summed_spin_channels and Spin.down in self.cohp:
            final_cohp = {}
            final_icohp = {}
            final_cohp[Spin.up] = np.sum([divided_cohp[Spin.up], divided_cohp[Spin.down]], axis=0)
            final_icohp[Spin.up] = np.sum([divided_icohp[Spin.up], divided_icohp[Spin.down]], axis=0)
        else:
            final_cohp = divided_cohp
            final_icohp = divided_icohp
        return Cohp(efermi=self.efermi, energies=self.energies, cohp=final_cohp, are_coops=self.are_coops, are_cobis=self.are_cobis, icohp=final_icohp)

    def get_summed_cohp_by_label_list(self, label_list, divisor=1, summed_spin_channels=False):
        """Returns a COHP object that includes a summed COHP divided by divisor.

        Args:
            label_list: list of labels for the COHP that should be included in the summed cohp
            divisor: float/int, the summed cohp will be divided by this divisor
            summed_spin_channels: bool, will sum the spin channels and return the sum in Spin.up if true

        Returns:
            Returns a COHP object including a summed COHP
        """
        first_cohpobject = self.get_cohp_by_label(label_list[0])
        summed_cohp = first_cohpobject.cohp.copy()
        summed_icohp = first_cohpobject.icohp.copy()
        for label in label_list[1:]:
            cohp_here = self.get_cohp_by_label(label)
            summed_cohp[Spin.up] = np.sum([summed_cohp[Spin.up], cohp_here.cohp[Spin.up]], axis=0)
            if Spin.down in summed_cohp:
                summed_cohp[Spin.down] = np.sum([summed_cohp[Spin.down], cohp_here.cohp[Spin.down]], axis=0)
            summed_icohp[Spin.up] = np.sum([summed_icohp[Spin.up], cohp_here.icohp[Spin.up]], axis=0)
            if Spin.down in summed_icohp:
                summed_icohp[Spin.down] = np.sum([summed_icohp[Spin.down], cohp_here.icohp[Spin.down]], axis=0)
        divided_cohp = {}
        divided_icohp = {}
        divided_cohp[Spin.up] = np.divide(summed_cohp[Spin.up], divisor)
        divided_icohp[Spin.up] = np.divide(summed_icohp[Spin.up], divisor)
        if Spin.down in summed_cohp:
            divided_cohp[Spin.down] = np.divide(summed_cohp[Spin.down], divisor)
            divided_icohp[Spin.down] = np.divide(summed_icohp[Spin.down], divisor)
        if summed_spin_channels and Spin.down in summed_cohp:
            final_cohp = {}
            final_icohp = {}
            final_cohp[Spin.up] = np.sum([divided_cohp[Spin.up], divided_cohp[Spin.down]], axis=0)
            final_icohp[Spin.up] = np.sum([divided_icohp[Spin.up], divided_icohp[Spin.down]], axis=0)
        else:
            final_cohp = divided_cohp
            final_icohp = divided_icohp
        return Cohp(efermi=first_cohpobject.efermi, energies=first_cohpobject.energies, cohp=final_cohp, are_coops=first_cohpobject.are_coops, are_cobis=first_cohpobject.are_coops, icohp=final_icohp)

    def get_summed_cohp_by_label_and_orbital_list(self, label_list, orbital_list, divisor=1, summed_spin_channels=False):
        """Returns a COHP object that includes a summed COHP divided by divisor.

        Args:
            label_list: list of labels for the COHP that should be included in the summed cohp
            orbital_list: list of orbitals for the COHPs that should be included in the summed cohp (same order as
                label_list)
            divisor: float/int, the summed cohp will be divided by this divisor
            summed_spin_channels: bool, will sum the spin channels and return the sum in Spin.up if true

        Returns:
            Returns a COHP object including a summed COHP
        """
        if not len(label_list) == len(orbital_list):
            raise ValueError("label_list and orbital_list don't have the same length!")
        first_cohpobject = self.get_orbital_resolved_cohp(label_list[0], orbital_list[0])
        summed_cohp = first_cohpobject.cohp.copy()
        summed_icohp = first_cohpobject.icohp.copy()
        for ilabel, label in enumerate(label_list[1:], start=1):
            cohp_here = self.get_orbital_resolved_cohp(label, orbital_list[ilabel])
            summed_cohp[Spin.up] = np.sum([summed_cohp[Spin.up], cohp_here.cohp.copy()[Spin.up]], axis=0)
            if Spin.down in summed_cohp:
                summed_cohp[Spin.down] = np.sum([summed_cohp[Spin.down], cohp_here.cohp.copy()[Spin.down]], axis=0)
            summed_icohp[Spin.up] = np.sum([summed_icohp[Spin.up], cohp_here.icohp.copy()[Spin.up]], axis=0)
            if Spin.down in summed_icohp:
                summed_icohp[Spin.down] = np.sum([summed_icohp[Spin.down], cohp_here.icohp.copy()[Spin.down]], axis=0)
        divided_cohp = {}
        divided_icohp = {}
        divided_cohp[Spin.up] = np.divide(summed_cohp[Spin.up], divisor)
        divided_icohp[Spin.up] = np.divide(summed_icohp[Spin.up], divisor)
        if Spin.down in summed_cohp:
            divided_cohp[Spin.down] = np.divide(summed_cohp[Spin.down], divisor)
            divided_icohp[Spin.down] = np.divide(summed_icohp[Spin.down], divisor)
        if summed_spin_channels and Spin.down in divided_cohp:
            final_cohp = {}
            final_icohp = {}
            final_cohp[Spin.up] = np.sum([divided_cohp[Spin.up], divided_cohp[Spin.down]], axis=0)
            final_icohp[Spin.up] = np.sum([divided_icohp[Spin.up], divided_icohp[Spin.down]], axis=0)
        else:
            final_cohp = divided_cohp
            final_icohp = divided_icohp
        return Cohp(efermi=first_cohpobject.efermi, energies=first_cohpobject.energies, cohp=final_cohp, are_coops=first_cohpobject.are_coops, are_cobis=first_cohpobject.are_cobis, icohp=final_icohp)

    def get_orbital_resolved_cohp(self, label, orbitals, summed_spin_channels=False):
        """Get orbital-resolved COHP.

        Args:
            label: bond label (Lobster: labels as in ICOHPLIST/ICOOPLIST.lobster).

            orbitals: The orbitals as a label, or list or tuple of the form
                [(n1, orbital1), (n2, orbital2)]. Orbitals can either be str,
                int, or Orbital.

            summed_spin_channels: bool, will sum the spin channels and return the sum in Spin.up if true

        Returns:
            A Cohp object if CompleteCohp contains orbital-resolved cohp,
            or None if it doesn't.

        Note: It currently assumes that orbitals are str if they aren't the
            other valid types. This is not ideal, but the easiest way to
            avoid unicode issues between python 2 and python 3.
        """
        if self.orb_res_cohp is None:
            return None
        if isinstance(orbitals, (list, tuple)):
            cohp_orbs = [d['orbitals'] for d in self.orb_res_cohp[label].values()]
            orbs = []
            for orbital in orbitals:
                if isinstance(orbital[1], int):
                    orbs.append((orbital[0], Orbital(orbital[1])))
                elif isinstance(orbital[1], Orbital):
                    orbs.append((orbital[0], orbital[1]))
                elif isinstance(orbital[1], str):
                    orbs.append((orbital[0], Orbital[orbital[1]]))
                else:
                    raise TypeError('Orbital must be str, int, or Orbital.')
            orb_index = cohp_orbs.index(orbs)
            orb_label = list(self.orb_res_cohp[label])[orb_index]
        elif isinstance(orbitals, str):
            orb_label = orbitals
        else:
            raise TypeError('Orbitals must be str, list, or tuple.')
        try:
            icohp = self.orb_res_cohp[label][orb_label]['ICOHP']
        except KeyError:
            icohp = None
        start_cohp = self.orb_res_cohp[label][orb_label]['COHP']
        start_icohp = icohp
        if summed_spin_channels and Spin.down in start_cohp:
            final_cohp = {}
            final_icohp = {}
            final_cohp[Spin.up] = np.sum([start_cohp[Spin.up], start_cohp[Spin.down]], axis=0)
            if start_icohp is not None:
                final_icohp[Spin.up] = np.sum([start_icohp[Spin.up], start_icohp[Spin.down]], axis=0)
        else:
            final_cohp = start_cohp
            final_icohp = start_icohp
        return Cohp(self.efermi, self.energies, final_cohp, icohp=final_icohp, are_coops=self.are_coops, are_cobis=self.are_cobis)

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Returns CompleteCohp object from dict representation."""
        cohp_dict = {}
        efermi = dct['efermi']
        energies = dct['energies']
        structure = Structure.from_dict(dct['structure'])
        are_cobis = dct.get('are_cobis', False)
        are_multi_center_cobis = dct.get('are_multi_center_cobis', False)
        are_coops = dct['are_coops']
        if 'bonds' in dct:
            bonds = {bond: {'length': dct['bonds'][bond]['length'], 'sites': tuple((PeriodicSite.from_dict(site) for site in dct['bonds'][bond]['sites'])), 'cells': dct['bonds'][bond].get('cells', None)} for bond in dct['bonds']}
        else:
            bonds = None
        for label in dct['COHP']:
            cohp = {Spin(int(spin)): np.array(dct['COHP'][label][spin]) for spin in dct['COHP'][label]}
            try:
                icohp = {Spin(int(spin)): np.array(dct['ICOHP'][label][spin]) for spin in dct['ICOHP'][label]}
            except KeyError:
                icohp = None
            if label == 'average':
                avg_cohp = Cohp(efermi, energies, cohp, icohp=icohp, are_coops=are_coops, are_cobis=are_cobis, are_multi_center_cobis=are_multi_center_cobis)
            else:
                cohp_dict[label] = Cohp(efermi, energies, cohp, icohp=icohp)
        if 'orb_res_cohp' in dct:
            orb_cohp: dict[str, dict] = {}
            for label in dct['orb_res_cohp']:
                orb_cohp[label] = {}
                for orb in dct['orb_res_cohp'][label]:
                    cohp = {Spin(int(s)): np.array(dct['orb_res_cohp'][label][orb]['COHP'][s], dtype=float) for s in dct['orb_res_cohp'][label][orb]['COHP']}
                    try:
                        icohp = {Spin(int(s)): np.array(dct['orb_res_cohp'][label][orb]['ICOHP'][s], dtype=float) for s in dct['orb_res_cohp'][label][orb]['ICOHP']}
                    except KeyError:
                        icohp = None
                    orbitals = [(int(o[0]), Orbital[o[1]]) for o in dct['orb_res_cohp'][label][orb]['orbitals']]
                    orb_cohp[label][orb] = {'COHP': cohp, 'ICOHP': icohp, 'orbitals': orbitals}
                if label not in dct['COHP'] or dct['COHP'][label] is None:
                    cohp = {Spin.up: np.sum(np.array([orb_cohp[label][orb]['COHP'][Spin.up] for orb in orb_cohp[label]]), axis=0)}
                    try:
                        cohp[Spin.down] = np.sum(np.array([orb_cohp[label][orb]['COHP'][Spin.down] for orb in orb_cohp[label]]), axis=0)
                    except KeyError:
                        pass
                orb_res_icohp = None in [orb_cohp[label][orb]['ICOHP'] for orb in orb_cohp[label]]
                if (label not in dct['ICOHP'] or dct['ICOHP'][label] is None) and orb_res_icohp:
                    icohp = {Spin.up: np.sum(np.array([orb_cohp[label][orb]['ICOHP'][Spin.up] for orb in orb_cohp[label]]), axis=0)}
                    try:
                        icohp[Spin.down] = np.sum(np.array([orb_cohp[label][orb]['ICOHP'][Spin.down] for orb in orb_cohp[label]]), axis=0)
                    except KeyError:
                        pass
        else:
            orb_cohp = {}
        are_cobis = dct.get('are_cobis', False)
        return cls(structure, avg_cohp, cohp_dict, bonds=bonds, are_coops=dct['are_coops'], are_cobis=are_cobis, are_multi_center_cobis=are_multi_center_cobis, orb_res_cohp=orb_cohp)

    @classmethod
    def from_file(cls, fmt, filename=None, structure_file=None, are_coops=False, are_cobis=False, are_multi_center_cobis=False) -> Self:
        """
        Creates a CompleteCohp object from an output file of a COHP
        calculation. Valid formats are either LMTO (for the Stuttgart
        LMTO-ASA code) or LOBSTER (for the LOBSTER code).

        Args:
            fmt: A string for the code that was used to calculate
                the COHPs so that the output file can be handled
                correctly. Can take the values "LMTO" or "LOBSTER".
            filename: Name of the COHP output file. Defaults to COPL
                for LMTO and COHPCAR.lobster/COOPCAR.lobster for LOBSTER.
            structure_file: Name of the file containing the structure.
                If no file name is given, use CTRL for LMTO and POSCAR
                for LOBSTER.
            are_coops: Indicates whether the populations are COOPs or
                COHPs. Defaults to False for COHPs.
            are_cobis: Indicates whether the populations are COBIs or
                COHPs. Defaults to False for COHPs.
            are_multi_center_cobis: Indicates whether this file
                includes information on multi-center COBIs

        Returns:
            A CompleteCohp object.
        """
        if are_coops and are_cobis:
            raise ValueError('You cannot have info about COOPs and COBIs in the same file.')
        fmt = fmt.upper()
        if fmt == 'LMTO':
            are_coops = False
            are_cobis = False
            orb_res_cohp = None
            if structure_file is None:
                structure_file = 'CTRL'
            if filename is None:
                filename = 'COPL'
            cohp_file: LMTOCopl | Cohpcar = LMTOCopl(filename=filename, to_eV=True)
        elif fmt == 'LOBSTER':
            if are_coops and are_cobis or (are_coops and are_multi_center_cobis) or (are_cobis and are_multi_center_cobis):
                raise ValueError('You cannot have info about COOPs, COBIs and/or multi-center COBIS in the same file.')
            if structure_file is None:
                structure_file = 'POSCAR'
            if filename is None and filename is None:
                if are_coops:
                    filename = 'COOPCAR.lobster'
                elif are_cobis or are_multi_center_cobis:
                    filename = 'COBICAR.lobster'
                else:
                    filename = 'COHPCAR.lobster'
            cohp_file = Cohpcar(filename=filename, are_coops=are_coops, are_cobis=are_cobis, are_multi_center_cobis=are_multi_center_cobis)
            orb_res_cohp = cohp_file.orb_res_cohp
        else:
            raise ValueError(f'Unknown format {fmt}. Valid formats are LMTO and LOBSTER.')
        structure = Structure.from_file(structure_file)
        efermi = cohp_file.efermi
        cohp_data = cohp_file.cohp_data
        energies = cohp_file.energies
        spins = [Spin.up, Spin.down] if cohp_file.is_spin_polarized else [Spin.up]
        if fmt == 'LOBSTER':
            energies += efermi
        if orb_res_cohp is not None:
            for label in orb_res_cohp:
                if cohp_file.cohp_data[label]['COHP'] is None:
                    cohp_data[label]['COHP'] = {sp: np.sum([orb_res_cohp[label][orbs]['COHP'][sp] for orbs in orb_res_cohp[label]], axis=0) for sp in spins}
                if cohp_file.cohp_data[label]['ICOHP'] is None:
                    cohp_data[label]['ICOHP'] = {sp: np.sum([orb_res_cohp[label][orbs]['ICOHP'][sp] for orbs in orb_res_cohp[label]], axis=0) for sp in spins}
        if fmt == 'LMTO':
            avg_data: dict[str, dict] = {'COHP': {}, 'ICOHP': {}}
            for i in avg_data:
                for spin in spins:
                    rows = np.array([v[i][spin] for v in cohp_data.values()])
                    avg = np.average(rows, axis=0)
                    avg_data[i].update({spin: np.array([round_to_sigfigs(a, 5) for a in avg], dtype=float)})
            avg_cohp = Cohp(efermi, energies, avg_data['COHP'], icohp=avg_data['ICOHP'])
        elif not are_multi_center_cobis:
            avg_cohp = Cohp(efermi, energies, cohp_data['average']['COHP'], icohp=cohp_data['average']['ICOHP'], are_coops=are_coops, are_cobis=are_cobis, are_multi_center_cobis=are_multi_center_cobis)
            del cohp_data['average']
        else:
            cohp = {}
            cohp[Spin.up] = np.array([np.array(c['COHP'][Spin.up]) for c in cohp_file.cohp_data.values() if len(c['sites']) <= 2]).mean(axis=0)
            try:
                cohp[Spin.down] = np.array([np.array(c['COHP'][Spin.down]) for c in cohp_file.cohp_data.values() if len(c['sites']) <= 2]).mean(axis=0)
            except KeyError:
                pass
            try:
                icohp = {}
                icohp[Spin.up] = np.array([np.array(c['ICOHP'][Spin.up]) for c in cohp_file.cohp_data.values() if len(c['sites']) <= 2]).mean(axis=0)
                try:
                    icohp[Spin.down] = np.array([np.array(c['ICOHP'][Spin.down]) for c in cohp_file.cohp_data.values() if len(c['sites']) <= 2]).mean(axis=0)
                except KeyError:
                    pass
            except KeyError:
                icohp = None
            avg_cohp = Cohp(efermi, energies, cohp, icohp=icohp, are_coops=are_coops, are_cobis=are_cobis, are_multi_center_cobis=are_multi_center_cobis)
        cohp_dict = {key: Cohp(efermi, energies, dct['COHP'], icohp=dct['ICOHP'], are_coops=are_coops, are_cobis=are_cobis, are_multi_center_cobis=are_multi_center_cobis) for key, dct in cohp_data.items()}
        bond_dict = {key: {'length': dct['length'], 'sites': [structure[site] for site in dct['sites']]} for key, dct in cohp_data.items()}
        return cls(structure, avg_cohp, cohp_dict, bonds=bond_dict, are_coops=are_coops, are_cobis=are_cobis, orb_res_cohp=orb_res_cohp)