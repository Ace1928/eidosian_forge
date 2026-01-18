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
class TestHarmonicVibrations:
    """Test the ase.vibrations.Vibrations object using a harmonic calculator
    """

    def setup(self):
        self.logfile = 'vibrations-log.txt'

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

    def test_harmonic_vibrations(self, testdir):
        """Check the numerics with a trivial case: one atom in harmonic well"""
        rng = np.random.RandomState(42)
        k = rng.rand()
        ref_atoms = Atoms('H', positions=np.zeros([1, 3]))
        atoms = ref_atoms.copy()
        mass = atoms.get_masses()[0]
        atoms.calc = ForceConstantCalculator(D=np.eye(3) * k, ref=ref_atoms, f0=np.zeros((1, 3)))
        vib = Vibrations(atoms, name='harmonic')
        vib.run()
        vib.read()
        expected_energy = units._hbar * np.sqrt(k * units._e * units.m ** 2 / mass / units._amu) / units._e
        assert np.allclose(vib.get_energies(), expected_energy)

    def test_consistency_with_vibrationsdata(self, testdir, random_dimer):
        vib = Vibrations(random_dimer, delta=1e-06, nfree=4)
        vib.run()
        vib_data = vib.get_vibrations()
        assert_array_almost_equal(vib.get_energies(), vib_data.get_energies())
        for mode_index in range(3 * len(vib.atoms)):
            assert_array_almost_equal(vib.get_mode(mode_index), vib_data.get_modes()[mode_index])
        assert_array_almost_equal(random_dimer.calc.D, vib_data.get_hessian_2d(), decimal=6)

    def test_json_manipulation(self, testdir, random_dimer):
        vib = Vibrations(random_dimer, name='interrupt')
        vib.run()
        disp_file = Path('interrupt/cache.1x-.json')
        comb_file = Path('interrupt/combined.json')
        assert disp_file.is_file()
        assert not comb_file.is_file()
        vib.split()
        assert vib.combine() == 13
        assert not disp_file.is_file()
        assert comb_file.is_file()
        with pytest.raises(RuntimeError):
            vib.run()
        vib.read()
        with open(disp_file, 'w') as fd:
            fd.write('hello')
        with pytest.raises(AssertionError):
            vib.split()
        os.remove(disp_file)
        vib.split()
        assert disp_file.is_file()
        assert not comb_file.is_file()

    def test_vibrations_methods(self, testdir, random_dimer):
        vib = Vibrations(random_dimer)
        vib.run()
        vib_energies = vib.get_energies()
        for image in vib.iterimages():
            assert len(image) == 2
        thermo = IdealGasThermo(vib_energies=vib_energies, geometry='linear', atoms=vib.atoms, symmetrynumber=2, spin=0)
        thermo.get_gibbs_energy(temperature=298.15, pressure=2 * 101325.0, verbose=False)
        with open(self.logfile, 'w') as fd:
            vib.summary(log=fd)
        with open(self.logfile, 'rt') as fd:
            log_txt = fd.read()
            assert log_txt == '\n'.join(VibrationsData._tabulate_from_energies(vib_energies)) + '\n'
        last_mode = vib.get_mode(-1)
        scale = 0.5
        assert_array_almost_equal(vib.show_as_force(-1, scale=scale, show=False).get_forces(), last_mode * 3 * len(vib.atoms) * scale)
        vib.write_mode(n=3, nimages=5)
        for i in range(3):
            assert not Path('vib.{}.traj'.format(i)).is_file()
        mode_traj = ase.io.read('vib.3.traj', index=':')
        assert len(mode_traj) == 5
        assert_array_almost_equal(mode_traj[0].get_all_distances(), random_dimer.get_all_distances())
        with pytest.raises(AssertionError):
            assert_array_almost_equal(mode_traj[4].get_all_distances(), random_dimer.get_all_distances())
        assert vib.clean(empty_files=True) == 0
        assert vib.clean() == 13
        assert len(list(vib.iterimages())) == 13
        d = dict(vib.iterdisplace(inplace=False))
        for name, image in vib.iterdisplace(inplace=True):
            assert d[name] == random_dimer

    def test_vibrations_restart_dir(self, testdir, random_dimer):
        vib = Vibrations(random_dimer)
        vib.run()
        freqs = vib.get_frequencies()
        assert freqs is not None
        atoms = random_dimer.copy()
        with ase.utils.workdir('run_from_here', mkdir=True):
            vib = Vibrations(atoms, name=str(Path.cwd().parent / 'vib'))
            assert_array_almost_equal(freqs, vib.get_frequencies())
            assert vib.clean() == 13