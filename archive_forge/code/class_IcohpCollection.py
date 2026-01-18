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
class IcohpCollection(MSONable):
    """Class to store IcohpValues.

    Attributes:
        are_coops (bool): Boolean to indicate if these are ICOOPs.
        are_cobis (bool): Boolean to indicate if these are ICOOPs.
        is_spin_polarized (bool): Boolean to indicate if the Lobster calculation was done spin polarized or not.
    """

    def __init__(self, list_labels, list_atom1, list_atom2, list_length, list_translation, list_num, list_icohp, is_spin_polarized, list_orb_icohp=None, are_coops=False, are_cobis=False) -> None:
        """
        Args:
            list_labels: list of labels for ICOHP/ICOOP values
            list_atom1: list of str of atomnames e.g. "O1"
            list_atom2: list of str of atomnames e.g. "O1"
            list_length: list of lengths of corresponding bonds in Angstrom
            list_translation: list of translation list, e.g. [0,0,0]
            list_num: list of equivalent bonds, usually 1 starting from Lobster 3.0.0
            list_icohp: list of dict={Spin.up: icohpvalue for spin.up, Spin.down: icohpvalue for spin.down}
            is_spin_polarized: Boolean to indicate if the Lobster calculation was done spin polarized or not Boolean to
                indicate if the Lobster calculation was done spin polarized or not
            list_orb_icohp: list of dict={[str(Orbital1)-str(Orbital2)]: {"icohp":{Spin.up: icohpvalue for spin.up,
                Spin.down: icohpvalue for spin.down}, "orbitals":[Orbital1, Orbital2]}}
            are_coops: Boolean to indicate whether ICOOPs are stored
            are_cobis: Boolean to indicate whether ICOBIs are stored.
        """
        if are_coops and are_cobis:
            raise ValueError('You cannot have info about COOPs and COBIs in the same file.')
        self._are_coops = are_coops
        self._are_cobis = are_cobis
        self._icohplist = {}
        self._is_spin_polarized = is_spin_polarized
        self._list_labels = list_labels
        self._list_atom1 = list_atom1
        self._list_atom2 = list_atom2
        self._list_length = list_length
        self._list_translation = list_translation
        self._list_num = list_num
        self._list_icohp = list_icohp
        self._list_orb_icohp = list_orb_icohp
        for ilist, listel in enumerate(list_labels):
            self._icohplist[listel] = IcohpValue(label=listel, atom1=list_atom1[ilist], atom2=list_atom2[ilist], length=list_length[ilist], translation=list_translation[ilist], num=list_num[ilist], icohp=list_icohp[ilist], are_coops=are_coops, are_cobis=are_cobis, orbitals=None if list_orb_icohp is None else list_orb_icohp[ilist])

    def __str__(self) -> str:
        joinstr = []
        for value in self._icohplist.values():
            joinstr.append(str(value))
        return '\n'.join(joinstr)

    def get_icohp_by_label(self, label, summed_spin_channels=True, spin=Spin.up, orbitals=None):
        """Get an icohp value for a certain bond as indicated by the label (bond labels starting by "1" as in
        ICOHPLIST/ICOOPLIST).

        Args:
            label: label in str format (usually the bond number in Icohplist.lobster/Icooplist.lobster
            summed_spin_channels: Boolean to indicate whether the ICOHPs/ICOOPs of both spin channels should be summed
            spin: if summed_spin_channels is equal to False, this spin indicates which spin channel should be returned
            orbitals: List of Orbital or "str(Orbital1)-str(Orbital2)"

        Returns:
            float describing ICOHP/ICOOP value
        """
        icohp_here = self._icohplist[label]
        if orbitals is None:
            if summed_spin_channels:
                return icohp_here.summed_icohp
            return icohp_here.icohpvalue(spin)
        if isinstance(orbitals, list):
            orbitals = f'{orbitals[0]}-{orbitals[1]}'
        if summed_spin_channels:
            return icohp_here.summed_orbital_icohp[orbitals]
        return icohp_here.icohpvalue_orbital(spin=spin, orbitals=orbitals)

    def get_summed_icohp_by_label_list(self, label_list, divisor=1.0, summed_spin_channels=True, spin=Spin.up):
        """Get the sum of several ICOHP values that are indicated by a list of labels
        (labels of the bonds are the same as in ICOHPLIST/ICOOPLIST).

        Args:
            label_list: list of labels of the ICOHPs/ICOOPs that should be summed
            divisor: is used to divide the sum
            summed_spin_channels: Boolean to indicate whether the ICOHPs/ICOOPs of both spin channels should be summed
            spin: if summed_spin_channels is equal to False, this spin indicates which spin channel should be returned

        Returns:
            float that is a sum of all ICOHPs/ICOOPs as indicated with label_list
        """
        sum_icohp = 0
        for label in label_list:
            icohp_here = self._icohplist[label]
            if icohp_here.num_bonds != 1:
                warnings.warn('One of the ICOHP values is an average over bonds. This is currently not considered.')
            if icohp_here._is_spin_polarized:
                if summed_spin_channels:
                    sum_icohp = sum_icohp + icohp_here.summed_icohp
                else:
                    sum_icohp = sum_icohp + icohp_here.icohpvalue(spin)
            else:
                sum_icohp = sum_icohp + icohp_here.icohpvalue(spin)
        return sum_icohp / divisor

    def get_icohp_dict_by_bondlengths(self, minbondlength=0.0, maxbondlength=8.0):
        """Get a dict of IcohpValues corresponding to certain bond lengths.

        Args:
            minbondlength: defines the minimum of the bond lengths of the bonds
            maxbondlength: defines the maximum of the bond lengths of the bonds.

        Returns:
            dict of IcohpValues, the keys correspond to the values from the initial list_labels.
        """
        new_icohp_dict = {}
        for value in self._icohplist.values():
            if value._length >= minbondlength and value._length <= maxbondlength:
                new_icohp_dict[value._label] = value
        return new_icohp_dict

    def get_icohp_dict_of_site(self, site, minsummedicohp=None, maxsummedicohp=None, minbondlength=0.0, maxbondlength=8.0, only_bonds_to=None):
        """Get a dict of IcohpValue for a certain site (indicated by integer).

        Args:
            site: integer describing the site of interest, order as in Icohplist.lobster/Icooplist.lobster, starts at 0
            minsummedicohp: float, minimal icohp/icoop of the bonds that are considered. It is the summed ICOHP value
                from both spin channels for spin polarized cases
            maxsummedicohp: float, maximal icohp/icoop of the bonds that are considered. It is the summed ICOHP value
                from both spin channels for spin polarized cases
            minbondlength: float, defines the minimum of the bond lengths of the bonds
            maxbondlength: float, defines the maximum of the bond lengths of the bonds
            only_bonds_to: list of strings describing the bonding partners that are allowed, e.g. ['O']

        Returns:
            dict of IcohpValues, the keys correspond to the values from the initial list_labels
        """
        new_icohp_dict = {}
        for key, value in self._icohplist.items():
            atomnumber1 = int(re.split('(\\d+)', value._atom1)[1]) - 1
            atomnumber2 = int(re.split('(\\d+)', value._atom2)[1]) - 1
            if site in (atomnumber1, atomnumber2):
                if site == atomnumber2:
                    save = value._atom1
                    value._atom1 = value._atom2
                    value._atom2 = save
                second_test = True if only_bonds_to is None else re.split('(\\d+)', value._atom2)[0] in only_bonds_to
                if value._length >= minbondlength and value._length <= maxbondlength and second_test:
                    if minsummedicohp is not None:
                        if value.summed_icohp >= minsummedicohp:
                            if maxsummedicohp is not None:
                                if value.summed_icohp <= maxsummedicohp:
                                    new_icohp_dict[key] = value
                            else:
                                new_icohp_dict[key] = value
                    elif maxsummedicohp is not None:
                        if value.summed_icohp <= maxsummedicohp:
                            new_icohp_dict[key] = value
                    else:
                        new_icohp_dict[key] = value
        return new_icohp_dict

    def extremum_icohpvalue(self, summed_spin_channels=True, spin=Spin.up):
        """Get ICOHP/ICOOP of strongest bond.

        Args:
            summed_spin_channels: Boolean to indicate whether the ICOHPs/ICOOPs of both spin channels should be summed.

            spin: if summed_spin_channels is equal to False, this spin indicates which spin channel should be returned

        Returns:
            lowest ICOHP/largest ICOOP value (i.e. ICOHP/ICOOP value of strongest bond)
        """
        extremum = -sys.float_info.max if self._are_coops or self._are_cobis else sys.float_info.max
        if not self._is_spin_polarized:
            if spin == Spin.down:
                warnings.warn('This spin channel does not exist. I am switching to Spin.up')
            spin = Spin.up
        for value in self._icohplist.values():
            if not value.is_spin_polarized or not summed_spin_channels:
                if not self._are_coops and (not self._are_cobis):
                    if value.icohpvalue(spin) < extremum:
                        extremum = value.icohpvalue(spin)
                elif value.icohpvalue(spin) > extremum:
                    extremum = value.icohpvalue(spin)
            elif not self._are_coops and (not self._are_cobis):
                if value.summed_icohp < extremum:
                    extremum = value.summed_icohp
            elif value.summed_icohp > extremum:
                extremum = value.summed_icohp
        return extremum

    @property
    def is_spin_polarized(self) -> bool:
        """Whether it is spin polarized."""
        return self._is_spin_polarized

    @property
    def are_coops(self) -> bool:
        """Whether this is a coop."""
        return self._are_coops

    @property
    def are_cobis(self) -> bool:
        """Whether this a cobi."""
        return self._are_cobis