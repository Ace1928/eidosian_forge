from __future__ import annotations
import re
from collections import defaultdict
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import CompleteDos, Dos
from pymatgen.io.feff import Header, Potential, Tags
class Xmu(MSONable):
    """
    Parser for data in 'xmu.dat' file.
    The file 'xmu.dat' contains XANES, EXAFS or NRIXS data depending on the
    situation; \\\\mu, \\\\mu_0, and \\\\chi = \\\\chi * \\\\mu_0/ \\\\mu_0/(edge+50eV) as
    functions of absolute energy E, relative energy E - E_f and wave number k.

    Default attributes:
        xmu: Photon absorption cross section of absorbing atom in material
        Energies: Energies of data point
        relative_energies: E - E_fermi
        wavenumber: k=\\\\sqrt(E -E_fermi)
        mu: The total absorption cross-section.
        mu0: The embedded atomic background absorption.
        chi: fine structure.
        Edge: Absorption Edge
        Absorbing atom: Species of absorbing atom
        Material: Formula of material
        Source: Source of structure
        Calculation: Type of Feff calculation performed
    """

    def __init__(self, header, parameters, absorbing_atom, data):
        """
        Args:
            header: Header object
            parameters: Tags object
            absorbing_atom (str/int): absorbing atom symbol or index
            data (numpy.ndarray, Nx6): cross_sections.
        """
        self.header = header
        self.parameters = parameters
        self.absorbing_atom = absorbing_atom
        self.data = np.array(data)

    @classmethod
    def from_file(cls, xmu_dat_file: str='xmu.dat', feff_inp_file: str='feff.inp') -> Self:
        """
        Get Xmu from file.

        Args:
            xmu_dat_file (str): filename and path for xmu.dat
            feff_inp_file (str): filename and path of feff.inp input file

        Returns:
            Xmu object
        """
        data = np.loadtxt(xmu_dat_file)
        header = Header.from_file(feff_inp_file)
        parameters = Tags.from_file(feff_inp_file)
        pots = Potential.pot_string_from_file(feff_inp_file)
        absorbing_atom = parameters['TARGET'] if 'RECIPROCAL' in parameters else pots.splitlines()[3].split()[2]
        return cls(header, parameters, absorbing_atom, data)

    @property
    def energies(self):
        """Returns the absolute energies in eV."""
        return self.data[:, 0]

    @property
    def relative_energies(self):
        """
        Returns energy with respect to the Fermi level.
        E - E_f.
        """
        return self.data[:, 1]

    @property
    def wavenumber(self):
        """
        Returns The wave number in units of \\\\AA^-1. k=\\\\sqrt(E - E_f) where E is
        the energy and E_f is the Fermi level computed from electron gas theory
        at the average interstitial charge density.
        """
        return self.data[:, 2]

    @property
    def mu(self):
        """Returns the total absorption cross-section."""
        return self.data[:, 3]

    @property
    def mu0(self):
        """Returns the embedded atomic background absorption."""
        return self.data[:, 4]

    @property
    def chi(self):
        """Returns the normalized fine structure."""
        return self.data[:, 5]

    @property
    def e_fermi(self):
        """Returns the Fermi level in eV."""
        return self.energies[0] - self.relative_energies[0]

    @property
    def source(self):
        """Returns source identification from Header file."""
        return self.header.source

    @property
    def calc(self):
        """Returns type of Feff calculation, XANES or EXAFS."""
        return 'XANES' if 'XANES' in self.parameters else 'EXAFS'

    @property
    def material_formula(self):
        """Returns chemical formula of material from feff.inp file."""
        try:
            form = self.header.formula
        except IndexError:
            form = 'No formula provided'
        return ''.join(map(str, form))

    @property
    def edge(self):
        """Returns excitation edge."""
        return self.parameters['EDGE']

    def as_dict(self):
        """Returns dict representations of Xmu object."""
        dct = MSONable.as_dict(self)
        dct['data'] = self.data.tolist()
        return dct