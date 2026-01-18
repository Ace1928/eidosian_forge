from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
class AimsOutCalcChunk(AimsOutChunk):
    """A part of the aims.out file corresponding to a single structure."""

    def __init__(self, lines: list[str], header: AimsOutHeaderChunk) -> None:
        """Construct the AimsOutCalcChunk.

        Args:
            lines (list[str]): The lines used for the structure
            header (.AimsOutHeaderChunk):  A summary of the relevant information from
                the aims.out header
        """
        super().__init__(lines)
        self._header = header.header_summary
        self._cache: dict[str, Any] = {}

    def _parse_structure(self) -> Structure | Molecule:
        """Parse a structure object from the file.

        For the given section of the aims output file generate the
        calculated structure.

        Returns:
            The structure or molecule for the calculation
        """
        species, coords, velocities, lattice = self._parse_lattice_atom_pos()
        site_properties: dict[str, Sequence[Any]] = dict()
        if len(velocities) > 0:
            site_properties['velocity'] = np.array(velocities)
        results = self.results
        site_prop_keys = {'forces': 'force', 'stresses': 'atomic_virial_stress', 'hirshfeld_charges': 'hirshfeld_charge', 'hirshfeld_volumes': 'hirshfeld_volume', 'hirshfeld_atomic_dipoles': 'hirshfeld_atomic_dipole'}
        properties = {prop: results[prop] for prop in results if prop not in site_prop_keys}
        for prop, site_key in site_prop_keys.items():
            if prop in results:
                site_properties[site_key] = results[prop]
        if lattice is not None:
            return Structure(lattice, species, coords, site_properties=site_properties, properties=properties, coords_are_cartesian=True)
        return Molecule(species, coords, site_properties=site_properties, properties=properties)

    def _parse_lattice_atom_pos(self) -> tuple[list[str], list[Vector3D], list[Vector3D], Lattice | None]:
        """Parse the lattice and atomic positions of the structure

        Returns:
            list[str]: The species symbols for the atoms in the structure
            list[Vector3D]: The Cartesian coordinates of the atoms
            list[Vector3D]: The velocities of the atoms
            Lattice or None: The Lattice for the structure
        """
        lattice_vectors = []
        velocities: list[Vector3D] = []
        species: list[str] = []
        coords: list[Vector3D] = []
        start_keys = ['Atomic structure (and velocities) as used in the preceding time step', 'Updated atomic structure', 'Atomic structure that was used in the preceding time step of the wrapper']
        line_start = self.reverse_search_for(start_keys)
        if line_start == LINE_NOT_FOUND:
            species = [sp.symbol for sp in self.initial_structure.species]
            coords = self.initial_structure.cart_coords.tolist()
            velocities = list(self.initial_structure.site_properties.get('velocity', []))
            lattice = self.initial_lattice
            return (species, coords, velocities, lattice)
        line_start += 1
        line_end = self.reverse_search_for(['Writing the current geometry to file "geometry.in.next_step"'], line_start)
        if line_end == LINE_NOT_FOUND:
            line_end = len(self.lines)
        for line in self.lines[line_start:line_end]:
            if 'lattice_vector   ' in line:
                lattice_vectors.append([float(inp) for inp in line.split()[1:]])
            elif 'atom   ' in line:
                line_split = line.split()
                species.append(line_split[4])
                coords.append([float(inp) for inp in line_split[1:4]])
            elif 'velocity   ' in line:
                velocities.append([float(inp) for inp in line.split()[1:]])
        lattice = Lattice(lattice_vectors) if len(lattice_vectors) == 3 else None
        return (species, coords, velocities, lattice)

    @property
    def species(self) -> list[str]:
        """The list of atomic symbols for all atoms in the structure"""
        if 'species' not in self._cache:
            self._cache['species'], self._cache['coords'], self._cache['velocities'], self._cache['lattice'] = self._parse_lattice_atom_pos()
        return self._cache['species']

    @property
    def coords(self) -> list[Vector3D]:
        """The cartesian coordinates of the atoms"""
        if 'coords' not in self._cache:
            self._cache['species'], self._cache['coords'], self._cache['velocities'], self._cache['lattice'] = self._parse_lattice_atom_pos()
        return self._cache['coords']

    @property
    def velocities(self) -> list[Vector3D]:
        """The velocities of the atoms"""
        if 'velocities' not in self._cache:
            self._cache['species'], self._cache['coords'], self._cache['velocities'], self._cache['lattice'] = self._parse_lattice_atom_pos()
        return self._cache['velocities']

    @property
    def lattice(self) -> Lattice:
        """The Lattice object for the structure"""
        if 'lattice' not in self._cache:
            self._cache['species'], self._cache['coords'], self._cache['velocities'], self._cache['lattice'] = self._parse_lattice_atom_pos()
        return self._cache['lattice']

    @property
    def forces(self) -> np.array[Vector3D] | None:
        """The forces from the aims.out file."""
        line_start = self.reverse_search_for(['Total atomic forces'])
        if line_start == LINE_NOT_FOUND:
            return None
        line_start += 1
        return np.array([[float(inp) for inp in line.split()[-3:]] for line in self.lines[line_start:line_start + self.n_atoms]])

    @property
    def stresses(self) -> np.array[Matrix3D] | None:
        """The stresses from the aims.out file and convert to kbar."""
        line_start = self.reverse_search_for(['Per atom stress (eV) used for heat flux calculation'])
        if line_start == LINE_NOT_FOUND:
            return None
        line_start += 3
        stresses = []
        for line in self.lines[line_start:line_start + self.n_atoms]:
            xx, yy, zz, xy, xz, yz = (float(d) for d in line.split()[2:8])
            stresses.append(Tensor.from_voigt([xx, yy, zz, yz, xz, xy]))
        return np.array(stresses) * EV_PER_A3_TO_KBAR

    @property
    def stress(self) -> Matrix3D | None:
        """The stress from the aims.out file and convert to kbar."""
        line_start = self.reverse_search_for(['Analytical stress tensor - Symmetrized', 'Numerical stress tensor'])
        if line_start == LINE_NOT_FOUND:
            return None
        stress = [[float(inp) for inp in line.split()[2:5]] for line in self.lines[line_start + 5:line_start + 8]]
        return np.array(stress) * EV_PER_A3_TO_KBAR

    @property
    def is_metallic(self) -> bool:
        """Is the system is metallic."""
        line_start = self.reverse_search_for(['material is metallic within the approximate finite broadening function (occupation_type)'])
        return line_start != LINE_NOT_FOUND

    @property
    def energy(self) -> float:
        """The energy from the aims.out file."""
        if self.initial_lattice is not None and self.is_metallic:
            line_ind = self.reverse_search_for(['Total energy corrected'])
        else:
            line_ind = self.reverse_search_for(['Total energy uncorrected'])
        if line_ind == LINE_NOT_FOUND:
            raise AimsParseError('No energy is associated with the structure.')
        return float(self.lines[line_ind].split()[5])

    @property
    def dipole(self) -> Vector3D | None:
        """The electric dipole moment from the aims.out file."""
        line_start = self.reverse_search_for(['Total dipole moment [eAng]'])
        if line_start == LINE_NOT_FOUND:
            return None
        line = self.lines[line_start]
        return np.array([float(inp) for inp in line.split()[6:9]])

    @property
    def dielectric_tensor(self) -> Matrix3D | None:
        """The dielectric tensor from the aims.out file."""
        line_start = self.reverse_search_for(['PARSE DFPT_dielectric_tensor'])
        if line_start == LINE_NOT_FOUND:
            return None
        lines = self.lines[line_start + 1:line_start + 4]
        return np.array([np.fromstring(line, sep=' ') for line in lines])

    @property
    def polarization(self) -> Vector3D | None:
        """The polarization vector from the aims.out file."""
        line_start = self.reverse_search_for(['| Cartesian Polarization'])
        if line_start == LINE_NOT_FOUND:
            return None
        line = self.lines[line_start]
        return np.array([float(s) for s in line.split()[-3:]])

    def _parse_homo_lumo(self) -> dict[str, float]:
        """Parse the HOMO/LUMO values and get band gap if periodic."""
        line_start = self.reverse_search_for(['Highest occupied state (VBM)'])
        homo = float(self.lines[line_start].split(' at ')[1].split('eV')[0].strip())
        line_start = self.reverse_search_for(['Lowest unoccupied state (CBM)'])
        lumo = float(self.lines[line_start].split(' at ')[1].split('eV')[0].strip())
        line_start = self.reverse_search_for(['verall HOMO-LUMO gap'])
        homo_lumo_gap = float(self.lines[line_start].split(':')[1].split('eV')[0].strip())
        line_start = self.reverse_search_for(['Smallest direct gap'])
        if line_start == LINE_NOT_FOUND:
            return {'vbm': homo, 'cbm': lumo, 'gap': homo_lumo_gap, 'direct_gap': homo_lumo_gap}
        direct_gap = float(self.lines[line_start].split(':')[1].split('eV')[0].strip())
        return {'vbm': homo, 'cbm': lumo, 'gap': homo_lumo_gap, 'direct_gap': direct_gap}

    def _parse_hirshfeld(self) -> None:
        """Parse the Hirshfled charges volumes, and dipole moments."""
        line_start = self.reverse_search_for(['Performing Hirshfeld analysis of fragment charges and moments.'])
        if line_start == LINE_NOT_FOUND:
            self._cache.update({'hirshfeld_charges': None, 'hirshfeld_volumes': None, 'hirshfeld_atomic_dipoles': None, 'hirshfeld_dipole': None})
            return
        line_inds = self.search_for_all('Hirshfeld charge', line_start, -1)
        hirshfeld_charges = np.array([float(self.lines[ind].split(':')[1]) for ind in line_inds])
        line_inds = self.search_for_all('Hirshfeld volume', line_start, -1)
        hirshfeld_volumes = np.array([float(self.lines[ind].split(':')[1]) for ind in line_inds])
        line_inds = self.search_for_all('Hirshfeld dipole vector', line_start, -1)
        hirshfeld_atomic_dipoles = np.array([[float(inp) for inp in self.lines[ind].split(':')[1].split()] for ind in line_inds])
        if self.lattice is None:
            hirshfeld_dipole = np.sum(hirshfeld_charges.reshape((-1, 1)) * self.coords, axis=1)
        else:
            hirshfeld_dipole = None
        self._cache.update({'hirshfeld_charges': hirshfeld_charges, 'hirshfeld_volumes': hirshfeld_volumes, 'hirshfeld_atomic_dipoles': hirshfeld_atomic_dipoles, 'hirshfeld_dipole': hirshfeld_dipole})

    @property
    def structure(self) -> Structure | Molecule:
        """The pytmagen SiteCollection of the chunk."""
        if 'structure' not in self._cache:
            self._cache['structure'] = self._parse_structure()
        return self._cache['structure']

    @property
    def results(self) -> dict[str, Any]:
        """Convert an AimsOutChunk to a Results Dictionary."""
        results = {'energy': self.energy, 'free_energy': self.free_energy, 'forces': self.forces, 'stress': self.stress, 'stresses': self.stresses, 'magmom': self.magmom, 'dipole': self.dipole, 'fermi_energy': self.E_f, 'n_iter': self.n_iter, 'hirshfeld_charges': self.hirshfeld_charges, 'hirshfeld_dipole': self.hirshfeld_dipole, 'hirshfeld_volumes': self.hirshfeld_volumes, 'hirshfeld_atomic_dipoles': self.hirshfeld_atomic_dipoles, 'dielectric_tensor': self.dielectric_tensor, 'polarization': self.polarization, 'vbm': self.vbm, 'cbm': self.cbm, 'gap': self.gap, 'direct_gap': self.direct_gap}
        return {key: value for key, value in results.items() if value is not None}

    @property
    def initial_structure(self) -> Structure | Molecule:
        """The initial structure for the calculation"""
        return self._header['initial_structure']

    @property
    def initial_lattice(self) -> Lattice | None:
        """The initial Lattice of the structure"""
        return self._header['initial_lattice']

    @property
    def n_atoms(self) -> int:
        """The number of atoms in the structure"""
        return self._header['n_atoms']

    @property
    def n_bands(self) -> int:
        """The number of Kohn-Sham states for the chunk."""
        return self._header['n_bands']

    @property
    def n_electrons(self) -> int:
        """The number of electrons for the chunk."""
        return self._header['n_electrons']

    @property
    def n_spins(self) -> int:
        """The number of spin channels for the chunk."""
        return self._header['n_spins']

    @property
    def electronic_temperature(self) -> float:
        """The electronic temperature for the chunk."""
        return self._header['electronic_temperature']

    @property
    def n_k_points(self) -> int:
        """The number of k_ppoints for the calculation."""
        return self._header['n_k_points']

    @property
    def k_points(self) -> Sequence[Vector3D]:
        """All k-points listed in the calculation."""
        return self._header['k_points']

    @property
    def k_point_weights(self) -> Sequence[float]:
        """The k-point weights for the calculation."""
        return self._header['k_point_weights']

    @property
    def free_energy(self) -> float | None:
        """The free energy of the calculation"""
        return self.parse_scalar('free_energy')

    @property
    def n_iter(self) -> int | None:
        """The number of steps needed to converge the SCF cycle for the chunk."""
        val = self.parse_scalar('number_of_iterations')
        if val is not None:
            return int(val)
        return None

    @property
    def magmom(self) -> float | None:
        """The magnetic moment of the structure"""
        return self.parse_scalar('magnetic_moment')

    @property
    def E_f(self) -> float | None:
        """The Fermi energy"""
        return self.parse_scalar('fermi_energy')

    @property
    def converged(self) -> bool:
        """True if the calculation is converged"""
        return len(self.lines) > 0 and 'Have a nice day.' in self.lines[-5:]

    @property
    def hirshfeld_charges(self) -> Sequence[float] | None:
        """The Hirshfeld charges of the system"""
        if 'hirshfeld_charges' not in self._cache:
            self._parse_hirshfeld()
        return self._cache['hirshfeld_charges']

    @property
    def hirshfeld_atomic_dipoles(self) -> Sequence[Vector3D] | None:
        """The Hirshfeld atomic dipoles of the system"""
        if 'hirshfeld_atomic_dipoles' not in self._cache:
            self._parse_hirshfeld()
        return self._cache['hirshfeld_atomic_dipoles']

    @property
    def hirshfeld_volumes(self) -> Sequence[float] | None:
        """The Hirshfeld atomic dipoles of the system"""
        if 'hirshfeld_volumes' not in self._cache:
            self._parse_hirshfeld()
        return self._cache['hirshfeld_volumes']

    @property
    def hirshfeld_dipole(self) -> None | Vector3D:
        """The Hirshfeld dipole of the system"""
        if 'hirshfeld_dipole' not in self._cache:
            self._parse_hirshfeld()
        return self._cache['hirshfeld_dipole']

    @property
    def vbm(self) -> float:
        """The valance band maximum"""
        return self._parse_homo_lumo()['vbm']

    @property
    def cbm(self) -> float:
        """The conduction band minimnum"""
        return self._parse_homo_lumo()['cbm']

    @property
    def gap(self) -> float:
        """The band gap"""
        return self._parse_homo_lumo()['gap']

    @property
    def direct_gap(self) -> float:
        """The direct bandgap"""
        return self._parse_homo_lumo()['direct_gap']