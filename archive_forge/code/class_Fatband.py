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
class Fatband:
    """
    Reads in FATBAND_x_y.lobster files.

    Attributes:
        efermi (float): Fermi energy read in from vasprun.xml.
        eigenvals (dict[Spin, np.ndarray]): Eigenvalues as a dictionary of numpy arrays of shape (nbands, nkpoints).
            The first index of the array refers to the band and the second to the index of the kpoint.
            The kpoints are ordered according to the order of the kpoints_array attribute.
            If the band structure is not spin polarized, we only store one data set under Spin.up.
        is_spin_polarized (bool): Boolean that tells you whether this was a spin-polarized calculation.
        kpoints_array (list[np.ndarray]): List of kpoints as numpy arrays, in frac_coords of the given
            lattice by default.
        label_dict (dict[str, Union[str, np.ndarray]]): Dictionary that links a kpoint (in frac coords or Cartesian
            coordinates depending on the coords attribute) to a label.
        lattice (Lattice): Lattice object of reciprocal lattice as read in from vasprun.xml.
        nbands (int): Number of bands used in the calculation.
        p_eigenvals (dict[Spin, np.ndarray]): Dictionary of orbital projections as {spin: array of dict}.
            The indices of the array are [band_index, kpoint_index].
            The dict is then built the following way: {"string of element": "string of orbital as read in
            from FATBAND file"}. If the band structure is not spin polarized, we only store one data set under Spin.up.
        structure (Structure): Structure read in from Structure object.
    """

    def __init__(self, filenames: str | list='.', kpoints_file: str='KPOINTS', vasprun_file: str | None='vasprun.xml', structure: Structure | IStructure | None=None, efermi: float | None=None):
        """
        Args:
            filenames (list or string): can be a list of file names or a path to a folder from which all
                "FATBAND_*" files will be read
            kpoints_file (str): KPOINTS file for bandstructure calculation, typically "KPOINTS".
            vasprun_file (str): Corresponding vasprun file.
                Instead, the Fermi energy from the DFT run can be provided. Then,
                this value should be set to None.
            structure (Structure): Structure object.
            efermi (float): fermi energy in eV
        """
        warnings.warn('Make sure all relevant FATBAND files were generated and read in!')
        warnings.warn('Use Lobster 3.2.0 or newer for fatband calculations!')
        if structure is None:
            raise ValueError('A structure object has to be provided')
        self.structure = structure
        if vasprun_file is None and efermi is None:
            raise ValueError('vasprun_file or efermi have to be provided')
        self.lattice = self.structure.lattice.reciprocal_lattice
        if vasprun_file is not None:
            self.efermi = Vasprun(filename=vasprun_file, ionic_step_skip=None, ionic_step_offset=0, parse_dos=True, parse_eigen=False, parse_projected_eigen=False, parse_potcar_file=False, occu_tol=1e-08, exception_on_bad_xml=True).efermi
        else:
            self.efermi = efermi
        kpoints_object = Kpoints.from_file(kpoints_file)
        atom_type = []
        atom_names = []
        orbital_names = []
        if not isinstance(filenames, list) or filenames is None:
            filenames_new = []
            if filenames is None:
                filenames = '.'
            for name in os.listdir(filenames):
                if fnmatch.fnmatch(name, 'FATBAND_*.lobster'):
                    filenames_new += [os.path.join(filenames, name)]
            filenames = filenames_new
        if len(filenames) == 0:
            raise ValueError('No FATBAND files in folder or given')
        for name in filenames:
            with zopen(name, mode='rt') as file:
                contents = file.read().split('\n')
            atom_names += [os.path.split(name)[1].split('_')[1].capitalize()]
            parameters = contents[0].split()
            atom_type += [re.split('[0-9]+', parameters[3])[0].capitalize()]
            orbital_names += [parameters[4]]
        atom_orbital_dict = {}
        for iatom, atom in enumerate(atom_names):
            if atom not in atom_orbital_dict:
                atom_orbital_dict[atom] = []
            atom_orbital_dict[atom] += [orbital_names[iatom]]
        for items in atom_orbital_dict.values():
            if len(set(items)) != len(items):
                raise ValueError('The are two FATBAND files for the same atom and orbital. The program will stop.')
            split = []
            for item in items:
                split += [item.split('_')[0]]
            for number in collections.Counter(split).values():
                if number not in (1, 3, 5, 7):
                    raise ValueError('Make sure all relevant orbitals were generated and that no duplicates (2p and 2p_x) are present')
        kpoints_array = []
        for ifilename, filename in enumerate(filenames):
            with zopen(filename, mode='rt') as file:
                contents = file.read().split('\n')
            if ifilename == 0:
                self.nbands = int(parameters[6])
                self.number_kpts = kpoints_object.num_kpts - int(contents[1].split()[2]) + 1
            if len(contents[1:]) == self.nbands + 2:
                self.is_spinpolarized = False
            elif len(contents[1:]) == self.nbands * 2 + 2:
                self.is_spinpolarized = True
            else:
                linenumbers = []
                for iline, line in enumerate(contents[1:self.nbands * 2 + 4]):
                    if line.split()[0] == '#':
                        linenumbers += [iline]
                if ifilename == 0:
                    self.is_spinpolarized = len(linenumbers) == 2
            if ifilename == 0:
                eigenvals = {}
                eigenvals[Spin.up] = [[collections.defaultdict(float) for _ in range(self.number_kpts)] for _ in range(self.nbands)]
                if self.is_spinpolarized:
                    eigenvals[Spin.down] = [[collections.defaultdict(float) for _ in range(self.number_kpts)] for _ in range(self.nbands)]
                p_eigenvals = {}
                p_eigenvals[Spin.up] = [[{str(elem): {str(orb): collections.defaultdict(float) for orb in atom_orbital_dict[elem]} for elem in atom_names} for _ in range(self.number_kpts)] for _ in range(self.nbands)]
                if self.is_spinpolarized:
                    p_eigenvals[Spin.down] = [[{str(elem): {str(orb): collections.defaultdict(float) for orb in atom_orbital_dict[elem]} for elem in atom_names} for _ in range(self.number_kpts)] for _ in range(self.nbands)]
            idx_kpt = -1
            linenumber = 0
            for line in contents[1:-1]:
                if line.split()[0] == '#':
                    KPOINT = np.array([float(line.split()[4]), float(line.split()[5]), float(line.split()[6])])
                    if ifilename == 0:
                        kpoints_array += [KPOINT]
                    linenumber = 0
                    iband = 0
                    idx_kpt += 1
                if linenumber == self.nbands:
                    iband = 0
                if line.split()[0] != '#':
                    if linenumber < self.nbands:
                        if ifilename == 0:
                            eigenvals[Spin.up][iband][idx_kpt] = float(line.split()[1]) + self.efermi
                        p_eigenvals[Spin.up][iband][idx_kpt][atom_names[ifilename]][orbital_names[ifilename]] = float(line.split()[2])
                    if linenumber >= self.nbands and self.is_spinpolarized:
                        if ifilename == 0:
                            eigenvals[Spin.down][iband][idx_kpt] = float(line.split()[1]) + self.efermi
                        p_eigenvals[Spin.down][iband][idx_kpt][atom_names[ifilename]][orbital_names[ifilename]] = float(line.split()[2])
                    linenumber += 1
                    iband += 1
        self.kpoints_array = kpoints_array
        self.eigenvals = eigenvals
        self.p_eigenvals = p_eigenvals
        label_dict = {}
        for idx, label in enumerate(kpoints_object.labels[-self.number_kpts:], start=0):
            if label is not None:
                label_dict[label] = kpoints_array[idx]
        self.label_dict = label_dict

    def get_bandstructure(self):
        """Returns a LobsterBandStructureSymmLine object which can be plotted with a normal BSPlotter."""
        return LobsterBandStructureSymmLine(kpoints=self.kpoints_array, eigenvals=self.eigenvals, lattice=self.lattice, efermi=self.efermi, labels_dict=self.label_dict, structure=self.structure, projections=self.p_eigenvals)