import re
import os
import numpy as np
import ase
from .vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator
class VaspDos:
    """Class for representing density-of-states produced by VASP

    The energies are in property self.energy

    Site-projected DOS is accesible via the self.site_dos method.

    Total and integrated DOS is accessible as numpy.ndarray's in the
    properties self.dos and self.integrated_dos. If the calculation is
    spin polarized, the arrays will be of shape (2, NDOS), else (1,
    NDOS).

    The self.efermi property contains the currently set Fermi
    level. Changing this value shifts the energies.

    """

    def __init__(self, doscar='DOSCAR', efermi=0.0):
        """Initialize"""
        self._efermi = 0.0
        self.read_doscar(doscar)
        self.efermi = efermi
        self.sort = []
        self.resort = []
        if os.path.isfile('ase-sort.dat'):
            file = open('ase-sort.dat', 'r')
            lines = file.readlines()
            file.close()
            for line in lines:
                data = line.split()
                self.sort.append(int(data[0]))
                self.resort.append(int(data[1]))

    def _set_efermi(self, efermi):
        """Set the Fermi level."""
        ef = efermi - self._efermi
        self._efermi = efermi
        self._total_dos[0, :] = self._total_dos[0, :] - ef
        try:
            self._site_dos[:, 0, :] = self._site_dos[:, 0, :] - ef
        except IndexError:
            pass

    def _get_efermi(self):
        return self._efermi
    efermi = property(_get_efermi, _set_efermi, None, 'Fermi energy.')

    def _get_energy(self):
        """Return the array with the energies."""
        return self._total_dos[0, :]
    energy = property(_get_energy, None, None, 'Array of energies')

    def site_dos(self, atom, orbital):
        """Return an NDOSx1 array with dos for the chosen atom and orbital.

        atom: int
            Atom index
        orbital: int or str
            Which orbital to plot

        If the orbital is given as an integer:
        If spin-unpolarized calculation, no phase factors:
        s = 0, p = 1, d = 2
        Spin-polarized, no phase factors:
        s-up = 0, s-down = 1, p-up = 2, p-down = 3, d-up = 4, d-down = 5
        If phase factors have been calculated, orbitals are
        s, py, pz, px, dxy, dyz, dz2, dxz, dx2
        double in the above fashion if spin polarized.

        """
        if self.resort:
            atom = self.resort[atom]
        if isinstance(orbital, int):
            return self._site_dos[atom, orbital + 1, :]
        n = self._site_dos.shape[1]
        from .vasp_data import PDOS_orbital_names_and_DOSCAR_column
        norb = PDOS_orbital_names_and_DOSCAR_column[n]
        return self._site_dos[atom, norb[orbital.lower()], :]

    def _get_dos(self):
        if self._total_dos.shape[0] == 3:
            return self._total_dos[1, :]
        elif self._total_dos.shape[0] == 5:
            return self._total_dos[1:3, :]
    dos = property(_get_dos, None, None, 'Average DOS in cell')

    def _get_integrated_dos(self):
        if self._total_dos.shape[0] == 3:
            return self._total_dos[2, :]
        elif self._total_dos.shape[0] == 5:
            return self._total_dos[3:5, :]
    integrated_dos = property(_get_integrated_dos, None, None, 'Integrated average DOS in cell')

    def read_doscar(self, fname='DOSCAR'):
        """Read a VASP DOSCAR file"""
        fd = open(fname)
        natoms = int(fd.readline().split()[0])
        [fd.readline() for nn in range(4)]
        ndos = int(fd.readline().split()[2])
        dos = []
        for nd in range(ndos):
            dos.append(np.array([float(x) for x in fd.readline().split()]))
        self._total_dos = np.array(dos).T
        dos = []
        for na in range(natoms):
            line = fd.readline()
            if line == '':
                break
            ndos = int(line.split()[2])
            line = fd.readline().split()
            cdos = np.empty((ndos, len(line)))
            cdos[0] = np.array(line)
            for nd in range(1, ndos):
                line = fd.readline().split()
                cdos[nd] = np.array([float(x) for x in line])
            dos.append(cdos.T)
        self._site_dos = np.array(dos)
        fd.close()