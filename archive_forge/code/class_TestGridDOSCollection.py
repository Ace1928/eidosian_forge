import pytest
from typing import Iterable
import numpy as np
from ase.spectrum.doscollection import (DOSCollection,
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData
class TestGridDOSCollection:

    @pytest.fixture
    def griddos(self):
        energies = np.linspace(1, 10, 7)
        weights = np.sin(energies)
        return GridDOSData(energies, weights, info={'my_key': 'my_value'})

    @pytest.fixture
    def another_griddos(self):
        energies = np.linspace(1, 10, 7)
        weights = np.cos(energies)
        return GridDOSData(energies, weights, info={'my_key': 'other_value'})

    @pytest.fixture
    def griddoscollection(self, griddos, another_griddos):
        return GridDOSCollection([griddos, another_griddos])

    def test_init_errors(self, griddos):
        with pytest.raises(TypeError):
            GridDOSCollection([RawDOSData([1.0], [1.0])])
        with pytest.raises(ValueError):
            energies = np.linspace(1, 10, 7) + 1
            GridDOSCollection([griddos, GridDOSData(energies, np.sin(energies))])
        with pytest.raises(ValueError):
            energies = np.linspace(1, 10, 6)
            GridDOSCollection([griddos, GridDOSData(energies, np.sin(energies))])
        with pytest.raises(ValueError):
            GridDOSCollection([], energies=None)
        with pytest.raises(ValueError):
            GridDOSCollection([griddos], energies=np.linspace(1, 10, 6))

    def test_select(self, griddos, another_griddos):
        gdc = GridDOSCollection([griddos, another_griddos])
        assert gdc.select(my_key='my_value')._almost_equals(GridDOSCollection([griddos]))
        assert gdc.select(my_key='not_present')._almost_equals(GridDOSCollection([], energies=griddos.get_energies()))
        assert gdc.select_not(my_key='my_value')._almost_equals(GridDOSCollection([another_griddos]))
        assert gdc.select(my_key='my_value').select_not(my_key='my_value')._almost_equals(GridDOSCollection([], energies=griddos.get_energies()))

    def test_sequence(self, griddos, another_griddos):
        gdc = GridDOSCollection([griddos, another_griddos])
        for i, (coll_dosdata, dosdata) in enumerate(zip(gdc, [griddos, another_griddos])):
            assert coll_dosdata._almost_equals(dosdata)
            assert gdc[i]._almost_equals(dosdata)

    def test_slicing(self, griddos, another_griddos):
        gdc = GridDOSCollection([griddos, another_griddos, griddos])
        assert gdc[1:]._almost_equals(GridDOSCollection([another_griddos, griddos]))
        assert gdc[:-1]._almost_equals(GridDOSCollection([griddos, another_griddos]))
        with pytest.raises(TypeError):
            gdc['string']

    @pytest.mark.parametrize('x, weights, info, error', [(np.linspace(1, 10, 12), [np.linspace(4, 1, 12), np.sin(range(12))], [{'entry': '1'}, {'entry': '2'}], None), (np.linspace(1, 5, 7), [np.sqrt(range(7))], [{'entry': '1'}], None), (np.linspace(1, 5, 7), [np.ones((3, 3))], None, IndexError), (np.linspace(1, 5, 7), np.array([]).reshape(0, 7), None, IndexError), (np.linspace(1, 5, 7), np.ones((2, 6)), None, IndexError)])
    def test_from_data(self, x, weights, info, error):
        if error is not None:
            with pytest.raises(error):
                dc = GridDOSCollection.from_data(x, weights, info=info)
        else:
            dc = GridDOSCollection.from_data(x, weights, info=info)
            for i, dos_data in enumerate(dc):
                assert dos_data.info == info[i]
                assert np.allclose(dos_data.get_energies(), x)
                assert np.allclose(dos_data.get_weights(), weights[i])

    @pytest.mark.usefixtures('figure')
    def test_plot_no_resample(self, griddoscollection, figure):
        ax = figure.add_subplot(111)
        griddoscollection.plot(ax=ax)
        assert np.allclose(ax.get_lines()[0].get_xdata(), griddoscollection[0].get_energies())
        assert np.allclose(ax.get_lines()[1].get_ydata(), griddoscollection[1].get_weights())