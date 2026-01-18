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
class TestVibrationsDataStaticMethods:

    @pytest.mark.parametrize('mask,expected_indices', [([True, True, False, True], [0, 1, 3]), ([False, False], []), ([], []), (np.array([True, True]), [0, 1]), (np.array([False, True, True]), [1, 2]), (np.array([], dtype=bool), [])])
    def test_indices_from_mask(self, mask, expected_indices):
        assert VibrationsData.indices_from_mask(mask) == expected_indices

    def test_tabulate_energies(self):
        energies = np.array([1.0, complex(2.0, 1.0), complex(1.0, 0.001)])
        table = VibrationsData._tabulate_from_energies(energies, im_tol=0.01)
        for sep_row in (0, 2, 6):
            assert table[sep_row] == '-' * 21
        assert tuple(table[1].strip().split()) == ('#', 'meV', 'cm^-1')
        expected_rows = [('0', '1000.0', '8065.5'), ('1', '1000.0i', '8065.5i'), ('2', '1000.0', '8065.5')]
        for row, expected in zip(table[3:6], expected_rows):
            assert tuple(row.split()) == expected
        assert table[7].split()[2] == '2.000'
        assert len(table) == 8
    na2 = Atoms('Na2', cell=[2, 2, 2], positions=[[0, 0, 0], [1, 1, 1]])
    na2_image_1 = na2.copy()
    na2_image_1.info.update({'mode#': '0', 'frequency_cm-1': 8065.5})
    na2_image_1.arrays['mode'] = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])

    @pytest.mark.parametrize('kwargs,expected', [(dict(atoms=na2, energies=[1.0], modes=np.array([[[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]])), [na2_image_1])])
    def test_get_jmol_images(self, kwargs, expected):
        from ase.calculators.calculator import compare_atoms
        jmol_images = list(VibrationsData._get_jmol_images(**kwargs))
        assert len(jmol_images) == len(expected)
        for image, reference in zip(jmol_images, expected):
            assert compare_atoms(image, reference) == []
            for key, value in reference.info.items():
                if key == 'frequency_cm-1':
                    assert float(image.info[key]) == pytest.approx(value, abs=0.1)
                else:
                    assert image.info[key] == value