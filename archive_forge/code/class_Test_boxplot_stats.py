from __future__ import annotations
import itertools
import pickle
from typing import Any
from unittest.mock import patch, Mock
from datetime import datetime, date, timedelta
import numpy as np
from numpy.testing import (assert_array_equal, assert_approx_equal,
import pytest
from matplotlib import _api, cbook
import matplotlib.colors as mcolors
from matplotlib.cbook import delete_masked_points, strip_math
class Test_boxplot_stats:

    def setup_method(self):
        np.random.seed(937)
        self.nrows = 37
        self.ncols = 4
        self.data = np.random.lognormal(size=(self.nrows, self.ncols), mean=1.5, sigma=1.75)
        self.known_keys = sorted(['mean', 'med', 'q1', 'q3', 'iqr', 'cilo', 'cihi', 'whislo', 'whishi', 'fliers', 'label'])
        self.std_results = cbook.boxplot_stats(self.data)
        self.known_nonbootstrapped_res = {'cihi': 6.816128326444485, 'cilo': -0.1489815330368689, 'iqr': 13.492709959447094, 'mean': 13.00447442387868, 'med': 3.333573396703808, 'fliers': np.array([92.55467075, 87.03819018, 42.23204914, 39.29390996]), 'q1': 1.3597529879465153, 'q3': 14.85246294739361, 'whishi': 27.89968824369963, 'whislo': 0.04214377496550292}
        self.known_bootstrapped_ci = {'cihi': 8.939577523357828, 'cilo': 1.8692703958676578}
        self.known_whis3_res = {'whishi': 42.23204913596987, 'whislo': 0.04214377496550292, 'fliers': np.array([92.55467075, 87.03819018])}
        self.known_res_percentiles = {'whislo': 0.1933685896907924, 'whishi': 42.23204913596987}
        self.known_res_range = {'whislo': 0.04214377496550292, 'whishi': 92.5546707521887}

    def test_form_main_list(self):
        assert isinstance(self.std_results, list)

    def test_form_each_dict(self):
        for res in self.std_results:
            assert isinstance(res, dict)

    def test_form_dict_keys(self):
        for res in self.std_results:
            assert set(res) <= set(self.known_keys)

    def test_results_baseline(self):
        res = self.std_results[0]
        for key, value in self.known_nonbootstrapped_res.items():
            assert_array_almost_equal(res[key], value)

    def test_results_bootstrapped(self):
        results = cbook.boxplot_stats(self.data, bootstrap=10000)
        res = results[0]
        for key, value in self.known_bootstrapped_ci.items():
            assert_approx_equal(res[key], value)

    def test_results_whiskers_float(self):
        results = cbook.boxplot_stats(self.data, whis=3)
        res = results[0]
        for key, value in self.known_whis3_res.items():
            assert_array_almost_equal(res[key], value)

    def test_results_whiskers_range(self):
        results = cbook.boxplot_stats(self.data, whis=[0, 100])
        res = results[0]
        for key, value in self.known_res_range.items():
            assert_array_almost_equal(res[key], value)

    def test_results_whiskers_percentiles(self):
        results = cbook.boxplot_stats(self.data, whis=[5, 95])
        res = results[0]
        for key, value in self.known_res_percentiles.items():
            assert_array_almost_equal(res[key], value)

    def test_results_withlabels(self):
        labels = ['Test1', 2, 'Aardvark', 4]
        results = cbook.boxplot_stats(self.data, labels=labels)
        for lab, res in zip(labels, results):
            assert res['label'] == lab
        results = cbook.boxplot_stats(self.data)
        for res in results:
            assert 'label' not in res

    def test_label_error(self):
        labels = [1, 2]
        with pytest.raises(ValueError):
            cbook.boxplot_stats(self.data, labels=labels)

    def test_bad_dims(self):
        data = np.random.normal(size=(34, 34, 34))
        with pytest.raises(ValueError):
            cbook.boxplot_stats(data)

    def test_boxplot_stats_autorange_false(self):
        x = np.zeros(shape=140)
        x = np.hstack([-25, x, 25])
        bstats_false = cbook.boxplot_stats(x, autorange=False)
        bstats_true = cbook.boxplot_stats(x, autorange=True)
        assert bstats_false[0]['whislo'] == 0
        assert bstats_false[0]['whishi'] == 0
        assert_array_almost_equal(bstats_false[0]['fliers'], [-25, 25])
        assert bstats_true[0]['whislo'] == -25
        assert bstats_true[0]['whishi'] == 25
        assert_array_almost_equal(bstats_true[0]['fliers'], [])