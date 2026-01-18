import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import IOContext
from ase.geometry import get_distances
from ase.cell import Cell
class ForceQMMM(Calculator):
    """
    Force-based QM/MM calculator

    QM forces are computed using a buffer region and then mixed abruptly
    with MM forces:

        F^i_QMMM = {   F^i_QM    if i in QM region
                   {   F^i_MM    otherwise

    cf. N. Bernstein, J. R. Kermode, and G. Csanyi,
    Rep. Prog. Phys. 72, 026501 (2009)
    and T. D. Swinburne and J. R. Kermode, Phys. Rev. B 96, 144102 (2017).
    """
    implemented_properties = ['forces', 'energy']

    def __init__(self, atoms, qm_selection_mask, qm_calc, mm_calc, buffer_width, vacuum=5.0, zero_mean=True, qm_cell_round_off=3, qm_radius=None):
        """
        ForceQMMM calculator

        Parameters:

        qm_selection_mask: list of ints, slice object or bool list/array
            Selection out of atoms that belong to the QM region.
        qm_calc: Calculator object
            QM-calculator.
        mm_calc: Calculator object
            MM-calculator (should be scaled, see :class:`RescaledCalculator`)
            Can use `ForceConstantCalculator` based on QM force constants, if
            available.
        vacuum: float or None
            Amount of vacuum to add around QM atoms.
        zero_mean: bool
            If True, add a correction to zero the mean force in each direction
        qm_cell_round_off: float
            Tolerance value in Angstrom to round the qm cluster cell
        qm_radius: 3x1 array of floats qm_radius for [x, y, z]
            3d qm radius for calculation of qm cluster cell. default is None
            and the radius is estimated from maximum distance between the atoms
            in qm region.
        """
        if len(atoms[qm_selection_mask]) == 0:
            raise ValueError('no QM atoms selected!')
        self.qm_selection_mask = qm_selection_mask
        self.qm_calc = qm_calc
        self.mm_calc = mm_calc
        self.vacuum = vacuum
        self.buffer_width = buffer_width
        self.zero_mean = zero_mean
        self.qm_cell_round_off = qm_cell_round_off
        self.qm_radius = qm_radius
        self.qm_buffer_mask = None
        Calculator.__init__(self)

    def initialize_qm_buffer_mask(self, atoms):
        """
        Initialises system to perform qm calculation
        """
        _, qm_distance_matrix = get_distances(atoms.positions[self.qm_selection_mask], atoms.positions, atoms.cell, atoms.pbc)
        self.qm_buffer_mask = np.zeros(len(atoms), dtype=bool)
        for r_qm in qm_distance_matrix:
            self.qm_buffer_mask[r_qm < self.buffer_width] = True

    def get_qm_cluster(self, atoms):
        if self.qm_buffer_mask is None:
            self.initialize_qm_buffer_mask(atoms)
        qm_cluster = atoms[self.qm_buffer_mask]
        del qm_cluster.constraints
        round_cell = False
        if self.qm_radius is None:
            round_cell = True
            R_qm, _ = get_distances(atoms.positions[self.qm_selection_mask], cell=atoms.cell, pbc=atoms.pbc)
            self.qm_radius = np.amax(np.amax(R_qm, axis=1), axis=0) * 0.5
        if atoms.cell.orthorhombic:
            cell_size = np.diagonal(atoms.cell)
        else:
            raise RuntimeError('NON-orthorhombic cell is not supported!')
        qm_cluster_pbc = atoms.pbc & (cell_size < 2.0 * (self.qm_radius + self.buffer_width))
        qm_cluster_cell = cell_size.copy()
        qm_cluster_cell[~qm_cluster_pbc] = 2.0 * (self.qm_radius[~qm_cluster_pbc] + self.buffer_width + self.vacuum)
        if round_cell:
            qm_cluster_cell[~qm_cluster_pbc] = np.round(qm_cluster_cell[~qm_cluster_pbc] / self.qm_cell_round_off) * self.qm_cell_round_off
        qm_cluster.set_cell(Cell(np.diag(qm_cluster_cell)))
        qm_cluster.pbc = qm_cluster_pbc
        qm_shift = 0.5 * qm_cluster.cell.diagonal() - qm_cluster.positions.mean(axis=0)
        if 'cell_origin' in qm_cluster.info:
            del qm_cluster.info['cell_origin']
        qm_cluster.positions[:, ~qm_cluster_pbc] += qm_shift[~qm_cluster_pbc]
        return qm_cluster

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        qm_cluster = self.get_qm_cluster(atoms)
        forces = self.mm_calc.get_forces(atoms)
        qm_forces = self.qm_calc.get_forces(qm_cluster)
        forces[self.qm_selection_mask] = qm_forces[self.qm_selection_mask[self.qm_buffer_mask]]
        if self.zero_mean:
            forces[:] -= forces.mean(axis=0)
        self.results['forces'] = forces
        self.results['energy'] = 0.0

    def get_region_from_masks(self, atoms=None, print_mapping=False):
        """
        creates region array from the masks of the calculators. The tags in
        the array are:
        QM - qm atoms
        buffer - buffer atoms
        MM - atoms treated with mm calculator
        """
        if atoms is None:
            if self.atoms is None:
                raise ValueError('Calculator has no atoms')
            else:
                atoms = self.atoms
        region = np.full_like(atoms, 'MM')
        region[self.qm_selection_mask] = np.full_like(region[self.qm_selection_mask], 'QM')
        buffer_only_mask = self.qm_buffer_mask & ~self.qm_selection_mask
        region[buffer_only_mask] = np.full_like(region[buffer_only_mask], 'buffer')
        if print_mapping:
            print(f'Mapping of {len(region):5d} atoms in total:')
            for region_id in np.unique(region):
                n_at = np.count_nonzero(region == region_id)
                print(f'{n_at:16d} {region_id}')
            qm_atoms = atoms[self.qm_selection_mask]
            symbol_counts = qm_atoms.symbols.formula.count()
            print('QM atoms types:')
            for symbol, count in symbol_counts.items():
                print(f'{count:16d} {symbol}')
        return region

    def set_masks_from_region(self, region):
        """
        Sets masks from provided region array
        """
        self.qm_selection_mask = region == 'QM'
        buffer_mask = region == 'buffer'
        self.qm_buffer_mask = self.qm_selection_mask ^ buffer_mask

    def export_extxyz(self, atoms=None, filename='qmmm_atoms.xyz'):
        """
        exports the atoms to extended xyz file with additional "region"
        array keeping the mapping between QM, buffer and MM parts of
        the simulation
        """
        if atoms is None:
            if self.atoms is None:
                raise ValueError('Calculator has no atoms')
            else:
                atoms = self.atoms
        region = self.get_region_from_masks(atoms=atoms)
        atoms_copy = atoms.copy()
        atoms_copy.new_array('region', region)
        atoms_copy.calc = self
        atoms_copy.write(filename, format='extxyz')

    @classmethod
    def import_extxyz(cls, filename, qm_calc, mm_calc):
        """
        A static method to import the the mapping from an estxyz file saved by
        export_extxyz() function
        Parameters
        ----------
        filename: string
            filename with saved configuration

        qm_calc: Calculator object
            QM-calculator.
        mm_calc: Calculator object
            MM-calculator (should be scaled, see :class:`RescaledCalculator`)
            Can use `ForceConstantCalculator` based on QM force constants, if
            available.

        Returns
        -------
        New object of ForceQMMM calculator with qm_selection_mask and
        qm_buffer_mask set according to the region array in the saved file
        """
        from ase.io import read
        atoms = read(filename, format='extxyz')
        if 'region' in atoms.arrays:
            region = atoms.get_array('region')
        else:
            raise RuntimeError("Please provide extxyz file with 'region' array")
        dummy_qm_mask = np.full_like(atoms, False, dtype=bool)
        dummy_qm_mask[0] = True
        self = cls(atoms, dummy_qm_mask, qm_calc, mm_calc, buffer_width=1.0)
        self.set_masks_from_region(region)
        return self