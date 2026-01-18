import os
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from ase import units, Atoms
import ase.io
from ase.calculators.qmmm import ForceConstantCalculator
from ase.vibrations import Vibrations, VibrationsData
from ase.thermochemistry import IdealGasThermo
class TestVibrationsData:

    @pytest.fixture
    def random_dimer(self):
        rng = np.random.RandomState(42)
        d = 1 + 0.5 * rng.rand()
        z_values = rng.randint(1, high=50, size=2)
        hessian = rng.rand(6, 6)
        hessian += hessian.T
        atoms = Atoms(z_values, [[0, 0, 0], [0, 0, d]])
        ref_atoms = atoms.copy()
        atoms.calc = ForceConstantCalculator(D=hessian, ref=ref_atoms, f0=np.zeros((2, 3)))
        return atoms

    @pytest.fixture
    def n2_data(self):
        return {'atoms': Atoms('N2', positions=[[0.0, 0.0, 0.05095057], [0.0, 0.0, 1.04904943]]), 'hessian': np.array([[[[0.00467554672, 0.0, 0.0], [-0.00467554672, 0.0, 0.0]], [[0.0, 0.00467554672, 0.0], [0.0, -0.00467554672, 0.0]], [[0.0, 0.0, 39.0392599], [0.0, 0.0, -39.0392599]]], [[[-0.00467554672, 0.0, 0.0], [0.00467554672, 0.0, 0.0]], [[0.0, -0.00467554672, 0.0], [0.0, 0.00467554672, 0.0]], [[0.0, 0.0, -39.0392599], [0.0, 0.0, 39.0392599]]]]), 'ref_frequencies': [0.0 + 0j, 6.0677553e-08 + 0j, 3.62010442e-06 + 0j, 13.4737571 + 0j, 13.4737571 + 0j, 1231.18496 + 0j], 'ref_zpe': 0.07799427233401508, 'ref_forces': np.array([[0.0, 0.0, -0.226722], [0.0, 0.0, 0.226722]])}

    @pytest.fixture
    def n2_unstable_data(self):
        return {'atoms': Atoms('N2', positions=[[0.0, 0.0, 0.45], [0.0, 0.0, -0.45]]), 'hessian': np.array([-5.150829928323684, 0.0, -0.6867385017096544, 5.150829928323684, 0.0, 0.6867385017096544, 0.0, -5.158454318599951, 0.0, 0.0, 5.158454318599951, 0.0, -0.6867385017096544, 0.0, 56.65107699250456, 0.6867385017096544, 0.0, -56.65107699250456, 5.150829928323684, 0.0, 0.6867385017096544, -5.150829928323684, 0.0, -0.6867385017096544, 0.0, 5.158454318599951, 0.0, 0.0, -5.158454318599951, 0.0, 0.6867385017096544, 0.0, -56.65107699250456, -0.6867385017096544, 0.0, 56.65107699250456]).reshape((2, 3, 2, 3))}

    @pytest.fixture
    def n2_vibdata(self, n2_data):
        return VibrationsData(n2_data['atoms'], n2_data['hessian'])

    def setup(self):
        self.jmol_file = 'vib-data.xyz'

    def test_init(self, n2_data):
        VibrationsData(n2_data['atoms'], n2_data['hessian'])

    def test_energies_and_modes(self, n2_data, n2_vibdata):
        energies, modes = n2_vibdata.get_energies_and_modes()
        assert_array_almost_equal(n2_data['ref_frequencies'], energies / units.invcm, decimal=5)
        assert_array_almost_equal(n2_data['ref_frequencies'], n2_vibdata.get_energies() / units.invcm, decimal=5)
        assert_array_almost_equal(n2_data['ref_frequencies'], n2_vibdata.get_frequencies(), decimal=5)
        assert n2_vibdata.get_zero_point_energy() == pytest.approx(n2_data['ref_zpe'])
        assert n2_vibdata.tabulate() == '\n'.join(VibrationsData._tabulate_from_energies(energies)) + '\n'
        atoms_with_forces = n2_vibdata.show_as_force(-1, show=False)
        try:
            assert_array_almost_equal(atoms_with_forces.get_forces(), n2_data['ref_forces'])
        except AssertionError:
            assert_array_almost_equal(atoms_with_forces.get_forces(), -n2_data['ref_forces'])

    def test_imaginary_energies(self, n2_unstable_data):
        vib_data = VibrationsData(n2_unstable_data['atoms'], n2_unstable_data['hessian'])
        assert vib_data.tabulate() == '\n'.join(VibrationsData._tabulate_from_energies(vib_data.get_energies())) + '\n'

    def test_zero_mass(self, n2_data):
        atoms = n2_data['atoms']
        atoms.set_masses([0.0, 1.0])
        vib_data = VibrationsData(atoms, n2_data['hessian'])
        with pytest.raises(ValueError):
            vib_data.get_energies_and_modes()

    def test_new_mass(self, n2_data, n2_vibdata):
        original_masses = n2_vibdata.get_atoms().get_masses()
        new_masses = original_masses * 3
        new_vib_data = n2_vibdata.with_new_masses(new_masses)
        assert_array_almost_equal(new_vib_data.get_atoms().get_masses(), new_masses)
        assert_array_almost_equal(n2_vibdata.get_energies() / np.sqrt(3), new_vib_data.get_energies())

    def test_fixed_atoms(self, n2_data):
        vib_data = VibrationsData(n2_data['atoms'], n2_data['hessian'][1:, :, 1:, :], indices=[1])
        assert vib_data.get_indices() == [1]
        assert vib_data.get_mask().tolist() == [False, True]

    def test_dos(self, n2_vibdata):
        with pytest.warns(np.ComplexWarning):
            dos = n2_vibdata.get_dos()
        assert_array_almost_equal(dos.get_energies(), n2_vibdata.get_energies())

    def test_pdos(self, n2_vibdata):
        with pytest.warns(np.ComplexWarning):
            pdos = n2_vibdata.get_pdos()
        assert_array_almost_equal(pdos[0].get_energies(), n2_vibdata.get_energies())
        assert_array_almost_equal(pdos[1].get_energies(), n2_vibdata.get_energies())
        assert sum(pdos[0].get_weights()) == pytest.approx(3.0)

    def test_todict(self, n2_data, n2_vibdata):
        vib_data_dict = n2_vibdata.todict()
        assert vib_data_dict['indices'] is None
        assert_array_almost_equal(vib_data_dict['atoms'].positions, n2_data['atoms'].positions)
        assert_array_almost_equal(vib_data_dict['hessian'], n2_data['hessian'])

    def test_dict_roundtrip(self, n2_vibdata):
        vib_data_dict = n2_vibdata.todict()
        vib_data_roundtrip = VibrationsData.fromdict(vib_data_dict)
        for getter in ('get_atoms',):
            assert getattr(n2_vibdata, getter)() == getattr(vib_data_roundtrip, getter)()
        for array_getter in ('get_hessian', 'get_hessian_2d', 'get_mask', 'get_indices'):
            assert_array_almost_equal(getattr(n2_vibdata, array_getter)(), getattr(vib_data_roundtrip, array_getter)())

    @pytest.mark.parametrize('indices, expected_mask', [([1], [False, True]), (None, [True, True])])
    def test_dict_indices(self, n2_vibdata, indices, expected_mask):
        vib_data_dict = n2_vibdata.todict()
        vib_data_dict['indices'] = indices
        if indices is not None:
            n_active = len(indices)
            vib_data_dict['hessian'] = np.asarray(vib_data_dict['hessian'])[:n_active, :, :n_active, :].tolist()
        vib_data_fromdict = VibrationsData.fromdict(vib_data_dict)
        assert_array_almost_equal(vib_data_fromdict.get_mask(), expected_mask)

    def test_jmol_roundtrip(self, testdir, n2_data):
        ir_intensities = np.random.RandomState(42).rand(6)
        vib_data = VibrationsData(n2_data['atoms'], n2_data['hessian'])
        vib_data.write_jmol(self.jmol_file, ir_intensities=ir_intensities)
        images = ase.io.read(self.jmol_file, index=':')
        for i, image in enumerate(images):
            assert_array_almost_equal(image.positions, vib_data.get_atoms().positions)
            assert image.info['IR_intensity'] == pytest.approx(ir_intensities[i])
            assert_array_almost_equal(image.arrays['mode'], vib_data.get_modes()[i])

    def test_bad_hessian(self, n2_data):
        bad_hessians = (None, 'fish', 1, np.array([1, 2, 3]), np.eye(6), np.array([[[1, 0, 0]], [[0, 0, 1]]]))
        for bad_hessian in bad_hessians:
            with pytest.raises(ValueError):
                VibrationsData(n2_data['atoms'], bad_hessian)

    def test_bad_hessian2d(self, n2_data):
        bad_hessians = (None, 'fish', 1, np.array([1, 2, 3]), n2_data['hessian'], np.array([[[1, 0, 0]], [[0, 0, 1]]]))
        for bad_hessian in bad_hessians:
            with pytest.raises(ValueError):
                VibrationsData.from_2d(n2_data['atoms'], bad_hessian)