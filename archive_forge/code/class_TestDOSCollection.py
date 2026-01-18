import pytest
from typing import Iterable
import numpy as np
from ase.spectrum.doscollection import (DOSCollection,
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData
class TestDOSCollection:

    @pytest.fixture
    def rawdos(self):
        return RawDOSData([1.0, 2.0, 4.0], [2.0, 3.0, 2.0], info={'my_key': 'my_value'})

    @pytest.fixture
    def another_rawdos(self):
        return RawDOSData([3.0, 2.0, 5.0], [1.0, 0.0, 2.0], info={'other_key': 'other_value'})

    @pytest.fixture
    def mindoscollection(self, rawdos, another_rawdos):
        return MinimalDOSCollection([rawdos, another_rawdos])

    @pytest.mark.parametrize('n_entries', [0, 1, 3])
    def test_sequence(self, rawdos, n_entries):
        dc = MinimalDOSCollection([rawdos] * n_entries)
        assert len(dc) == n_entries
        for i in range(n_entries):
            assert dc[i] == rawdos
        with pytest.raises(IndexError):
            dc[n_entries + 1]
        with pytest.raises(TypeError):
            dc['hello']
    linewidths = [1, 5, None]

    @pytest.mark.usefixtures('figure')
    @pytest.mark.parametrize('linewidth', linewidths)
    def test_plot(self, mindoscollection, figure, linewidth):
        npts = 20
        if linewidth is None:
            mplargs = None
        else:
            mplargs = {'linewidth': linewidth}
        ax = figure.add_subplot(111)
        with pytest.warns(UserWarning):
            ax_out = mindoscollection.plot(npts=npts, ax=ax, mplargs=mplargs)
        assert ax_out == ax
        assert [line.get_label() for line in ax.get_legend().get_lines()] == ['my_key: my_value', 'other_key: other_value']

    def test_slicing(self, rawdos, another_rawdos):
        dc = MinimalDOSCollection([rawdos, another_rawdos, rawdos])
        assert dc[1:]._almost_equals(MinimalDOSCollection([another_rawdos, rawdos]))
        assert dc[:-1]._almost_equals(MinimalDOSCollection([rawdos, another_rawdos]))

    def test_collection_equality(self, rawdos, another_rawdos):
        equality_data = [([], [], True), ([rawdos], [rawdos], True), ([rawdos, another_rawdos], [rawdos, another_rawdos], True), ([], [rawdos], False), ([rawdos], [], False), ([rawdos, another_rawdos], [rawdos], False), ([rawdos, another_rawdos], [another_rawdos, rawdos], False)]
        for series_1, series_2, isequal in equality_data:
            assert MinimalDOSCollection(series_1)._almost_equals(MinimalDOSCollection(series_2)) == isequal

    @pytest.mark.parametrize('other', [True, 1, 0.5, 'string', rawdos])
    def test_equality_wrongtype(self, rawdos, other):
        assert not MinimalDOSCollection([rawdos])._almost_equals(other)

    def test_addition(self, rawdos, another_rawdos):
        dc = MinimalDOSCollection([rawdos])
        double_dc = dc + dc
        assert len(double_dc) == 2
        assert double_dc[0]._almost_equals(rawdos)
        assert double_dc[1]._almost_equals(rawdos)
        assert (dc + MinimalDOSCollection([another_rawdos]))._almost_equals(dc + another_rawdos)
        with pytest.raises(TypeError):
            MinimalDOSCollection([rawdos]) + YetAnotherDOSCollection([rawdos])
        with pytest.raises(TypeError):
            MinimalDOSCollection([rawdos]) + 'string'

    @pytest.mark.parametrize('options', [{'energies': [1.0, 1.1, 1.2], 'width': 1.3, 'smearing': 'Gauss'}, {'energies': [1.7, 2.1, 2.0], 'width': 3.4, 'smearing': 'Gauss'}])
    def test_sample(self, rawdos, another_rawdos, options):
        dc = MinimalDOSCollection([rawdos, another_rawdos])
        sampled_data = dc._sample(**options)
        for i, data in enumerate((rawdos, another_rawdos)):
            newdos_weights = data._sample(**options)
            assert np.allclose(sampled_data[i, :], newdos_weights)
            assert np.all(sampled_data)
    sample_grid_options = [{'npts': 10, 'xmin': -2, 'xmax': 10, 'padding': 3, 'width': 1}, {'npts': 12, 'xmin': 0, 'xmax': 4, 'padding': 2.1, 'width': 2.3}]

    @pytest.mark.parametrize('options', sample_grid_options)
    def test_sample_grid(self, rawdos, another_rawdos, options):
        ref_min = min(rawdos.get_energies())
        ref_max = max(another_rawdos.get_energies())
        dc = MinimalDOSCollection([rawdos, another_rawdos])
        dos = dc.sample_grid(10, xmax=options['xmax'], padding=options['padding'], width=options['width'])
        energies = dos.get_energies()
        assert pytest.approx(energies[0]) == ref_min - options['padding'] * options['width']
        assert pytest.approx(energies[-1]) == options['xmax']
        dos = dc.sample_grid(10, xmin=options['xmin'], padding=options['padding'], width=options['width'])
        energies = dos.get_energies()
        assert pytest.approx(energies[0]) == options['xmin']
        assert pytest.approx(energies[-1]) == ref_max + options['padding'] * options['width']
        dos = dc.sample_grid(**options)
        energies = dos.get_energies()
        weights = dos.get_all_weights()
        for i, data in enumerate((rawdos, another_rawdos)):
            tmp_dos = data.sample_grid(**options)
            tmp_weights = tmp_dos.get_weights()
            assert np.allclose(weights[i, :], tmp_weights)

    def test_sample_empty(self):
        empty_dc = MinimalDOSCollection([])
        with pytest.raises(IndexError):
            empty_dc._sample(10)
        with pytest.raises(IndexError):
            empty_dc.sample_grid(10)

    @pytest.mark.parametrize('x, weights, bad_info', [([1, 2, 4, 5], [[0, 1, 1, 0], [2, 1, 2, 1]], [{'notenough': 'entries'}]), ([3.1, 2.4, 1.1], [[2, 1.0, 3.12]], [{'too': 'many'}, {'entries': 'here'}])])
    def test_from_data(self, x, weights, bad_info):
        dc = DOSCollection.from_data(x, weights)
        for i, dos_data in enumerate(dc):
            assert dos_data.info == {}
            assert np.allclose(dos_data.get_energies(), x)
            assert np.allclose(dos_data.get_weights(), weights[i])
        with pytest.raises(ValueError):
            dc = DOSCollection.from_data(x, weights, info=bad_info)
    collection_data = [[([1.0, 2.0, 3.0], [1.0, 1.0, 2.0])], [([1.0, 2.0, 3.0], [1.0, 1.0, 2.0]), ([2.0, 3.5], [0.5, 1.0])], [([1.0, 2.0, 3.0], [1.0, 1.0, 2.0]), ([2.0, 3.5], [0.5, 1.0]), ([1.0], [0.25])]]
    collection_info = [[{'el': 'C', 'index': '1'}], [{'el': 'C', 'index': '1'}, {'el': 'C', 'index': '2'}], [{'el': 'C', 'index': '1'}, {'el': 'C', 'index': '2'}, {'el': 'C', 'index': '2'}]]
    expected_sum = [([1.0, 2.0, 3.0], [1.0, 1.0, 2.0], {'el': 'C', 'index': '1'}), ([1.0, 2.0, 3.0, 2.0, 3.5], [1.0, 1.0, 2.0, 0.5, 1.0], {'el': 'C'}), ([1.0, 2.0, 3.0, 2.0, 3.5, 1.0], [1.0, 1.0, 2.0, 0.5, 1.0, 0.25], {'el': 'C'})]

    @pytest.mark.parametrize('collection_data, collection_info, expected', zip(collection_data, collection_info, expected_sum))
    def test_sum_all(self, collection_data, collection_info, expected):
        dc = DOSCollection([RawDOSData(*item, info=info) for item, info in zip(collection_data, collection_info)])
        summed_dc = dc.sum_all()
        energies, weights, ref_info = expected
        assert np.allclose(summed_dc.get_energies(), energies)
        assert np.allclose(summed_dc.get_weights(), weights)
        assert summed_dc.info == ref_info

    def test_sum_empty(self):
        dc = DOSCollection([])
        with pytest.raises(IndexError):
            dc.sum_all()

    @pytest.mark.parametrize('collection_data, collection_info', zip(collection_data, collection_info))
    def test_total(self, collection_data, collection_info):
        dc = DOSCollection([RawDOSData(*item, info=info) for item, info in zip(collection_data, collection_info)])
        summed = dc.sum_all()
        total = dc.total()
        assert np.allclose(summed.get_energies(), total.get_energies())
        assert np.allclose(summed.get_weights(), total.get_weights())
        assert set(total.info.items()) - set(summed.info.items()) == set([('label', 'Total')])
    select_info = [[{'a': '1', 'b': '1'}, {'a': '2'}], [{'a': '1', 'b': '1'}, {'a': '1', 'b': '2'}], [{'a': '1'}, {'a': '2'}], [{'a': '1', 'b': '1', 'c': '1'}, {'a': '1', 'b': '1', 'c': '2'}, {'a': '1', 'b': '2', 'c': '3'}]]
    select_query = [{'a': '1'}, {'a': '1'}, {'a': '0'}, {'a': '1', 'b': '1'}]
    select_result = [[{'a': '1', 'b': '1'}], [{'a': '1', 'b': '1'}, {'a': '1', 'b': '2'}], None, [{'a': '1', 'b': '1', 'c': '1'}, {'a': '1', 'b': '1', 'c': '2'}]]
    select_not_result = [[{'a': '2'}], None, [{'a': '1'}, {'a': '2'}], [{'a': '1', 'b': '2', 'c': '3'}]]
    sum_by_result = [[{'a': '1', 'b': '1'}, {'a': '2'}], [{'a': '1'}], [{'a': '1'}, {'a': '2'}], [{'a': '1', 'b': '1'}, {'a': '1', 'b': '2', 'c': '3'}]]

    @pytest.mark.parametrize('select_info, select_query, select_result, select_not_result, sum_by_result', zip(select_info, select_query, select_result, select_not_result, sum_by_result))
    def test_select(self, select_info, select_query, select_result, select_not_result, sum_by_result):
        dc = DOSCollection([RawDOSData([0.0], [0.0], info=info) for info in select_info])
        if select_result is None:
            assert dc.select(**select_query)._almost_equals(DOSCollection([]))
        else:
            assert select_result == [data.info for data in dc.select(**select_query)]
        if select_not_result is None:
            assert dc.select_not(**select_query)._almost_equals(DOSCollection([]))
        else:
            assert select_not_result == [data.info for data in dc.select_not(**select_query)]
        assert sum_by_result == [data.info for data in dc.sum_by(*sorted(select_query.keys()))]