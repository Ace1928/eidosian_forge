from __future__ import annotations
import collections
import fnmatch
import os
import re
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import LobsterBandStructureSymmLine
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.dos import Dos, LobsterCompleteDos
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Vasprun, VolumetricData
from pymatgen.util.due import Doi, due
class Doscar:
    """
    Class to deal with Lobster's projected DOS and local projected DOS.
    The beforehand quantum-chemical calculation was performed with VASP.

    Attributes:
        completedos (LobsterCompleteDos): LobsterCompleteDos Object.
        pdos (list): List of Dict including numpy arrays with pdos. Access as
            pdos[atomindex]['orbitalstring']['Spin.up/Spin.down'].
        tdos (Dos): Dos Object of the total density of states.
        energies (numpy.ndarray): Numpy array of the energies at which the DOS was calculated
            (in eV, relative to Efermi).
        tdensities (dict): tdensities[Spin.up]: numpy array of the total density of states for
            the Spin.up contribution at each of the energies. tdensities[Spin.down]: numpy array
            of the total density of states for the Spin.down contribution at each of the energies.
            If is_spin_polarized=False, tdensities[Spin.up]: numpy array of the total density of states.
        itdensities (dict): itdensities[Spin.up]: numpy array of the total density of states for
            the Spin.up contribution at each of the energies. itdensities[Spin.down]: numpy array
            of the total density of states for the Spin.down contribution at each of the energies.
            If is_spin_polarized=False, itdensities[Spin.up]: numpy array of the total density of states.
        is_spin_polarized (bool): Boolean. Tells if the system is spin polarized.
    """

    def __init__(self, doscar: str='DOSCAR.lobster', structure_file: str | None='POSCAR', structure: IStructure | Structure | None=None):
        """
        Args:
            doscar: DOSCAR filename, typically "DOSCAR.lobster"
            structure_file: for vasp, this is typically "POSCAR"
            structure: instead of a structure file, the structure can be given
                directly. structure_file will be preferred.
        """
        self._doscar = doscar
        self._final_structure = Structure.from_file(structure_file) if structure_file is not None else structure
        self._parse_doscar()

    def _parse_doscar(self):
        doscar = self._doscar
        tdensities = {}
        itdensities = {}
        with zopen(doscar, mode='rt') as file:
            n_atoms = int(file.readline().split()[0])
            efermi = float([file.readline() for nn in range(4)][3].split()[17])
            dos = []
            orbitals = []
            for _atom in range(n_atoms + 1):
                line = file.readline()
                ndos = int(line.split()[2])
                orbitals += [line.split(';')[-1].split()]
                line = file.readline().split()
                cdos = np.zeros((ndos, len(line)))
                cdos[0] = np.array(line)
                for nd in range(1, ndos):
                    line = file.readline().split()
                    cdos[nd] = np.array(line)
                dos += [cdos]
        doshere = np.array(dos[0])
        if len(doshere[0, :]) == 5:
            self._is_spin_polarized = True
        elif len(doshere[0, :]) == 3:
            self._is_spin_polarized = False
        else:
            raise ValueError("There is something wrong with the DOSCAR. Can't extract spin polarization.")
        energies = doshere[:, 0]
        if not self._is_spin_polarized:
            tdensities[Spin.up] = doshere[:, 1]
            itdensities[Spin.up] = doshere[:, 2]
            pdoss = []
            spin = Spin.up
            for atom in range(n_atoms):
                pdos = defaultdict(dict)
                data = dos[atom + 1]
                _, ncol = data.shape
                orbnumber = 0
                for j in range(1, ncol):
                    orb = orbitals[atom + 1][orbnumber]
                    pdos[orb][spin] = data[:, j]
                    orbnumber = orbnumber + 1
                pdoss += [pdos]
        else:
            tdensities[Spin.up] = doshere[:, 1]
            tdensities[Spin.down] = doshere[:, 2]
            itdensities[Spin.up] = doshere[:, 3]
            itdensities[Spin.down] = doshere[:, 4]
            pdoss = []
            for atom in range(n_atoms):
                pdos = defaultdict(dict)
                data = dos[atom + 1]
                _, ncol = data.shape
                orbnumber = 0
                for j in range(1, ncol):
                    spin = Spin.down if j % 2 == 0 else Spin.up
                    orb = orbitals[atom + 1][orbnumber]
                    pdos[orb][spin] = data[:, j]
                    if j % 2 == 0:
                        orbnumber = orbnumber + 1
                pdoss += [pdos]
        self._efermi = efermi
        self._pdos = pdoss
        self._tdos = Dos(efermi, energies, tdensities)
        self._energies = energies
        self._tdensities = tdensities
        self._itdensities = itdensities
        final_struct = self._final_structure
        pdossneu = {final_struct[i]: pdos for i, pdos in enumerate(self._pdos)}
        self._completedos = LobsterCompleteDos(final_struct, self._tdos, pdossneu)

    @property
    def completedos(self) -> LobsterCompleteDos:
        """LobsterCompleteDos"""
        return self._completedos

    @property
    def pdos(self) -> list:
        """Projected DOS"""
        return self._pdos

    @property
    def tdos(self) -> Dos:
        """Total DOS"""
        return self._tdos

    @property
    def energies(self) -> np.ndarray:
        """Energies"""
        return self._energies

    @property
    def tdensities(self) -> dict[Spin, np.ndarray]:
        """total densities as a np.ndarray"""
        return self._tdensities

    @property
    def itdensities(self) -> dict[Spin, np.ndarray]:
        """integrated total densities as a np.ndarray"""
        return self._itdensities

    @property
    def is_spin_polarized(self) -> bool:
        """Whether run is spin polarized."""
        return self._is_spin_polarized