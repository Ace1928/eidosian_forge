import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
class TestCdist:

    def setup_method(self):
        self.rnd_eo_names = ['random-float32-data', 'random-int-data', 'random-uint-data', 'random-double-data', 'random-bool-data']
        self.valid_upcasts = {'bool': [np_ulong, np_long, np.float32, np.float64], 'uint': [np_long, np.float32, np.float64], 'int': [np.float32, np.float64], 'float32': [np.float64]}

    def test_cdist_extra_args(self, metric):
        X1 = [[1.0, 2.0, 3.0], [1.2, 2.3, 3.4], [2.2, 2.3, 4.4]]
        X2 = [[7.0, 5.0, 8.0], [7.5, 5.8, 8.4], [5.5, 5.8, 4.4]]
        kwargs = {'N0tV4l1D_p4raM': 3.14, 'w': np.arange(3)}
        args = [3.14] * 200
        with pytest.raises(TypeError):
            cdist(X1, X2, metric=metric, **kwargs)
        with pytest.raises(TypeError):
            cdist(X1, X2, metric=eval(metric), **kwargs)
        with pytest.raises(TypeError):
            cdist(X1, X2, metric='test_' + metric, **kwargs)
        with pytest.raises(TypeError):
            cdist(X1, X2, *args, metric=metric)
        with pytest.raises(TypeError):
            cdist(X1, X2, *args, metric=eval(metric))
        with pytest.raises(TypeError):
            cdist(X1, X2, *args, metric='test_' + metric)

    def test_cdist_extra_args_custom(self):

        def _my_metric(x, y, arg, kwarg=1, kwarg2=2):
            return arg + kwarg + kwarg2
        X1 = [[1.0, 2.0, 3.0], [1.2, 2.3, 3.4], [2.2, 2.3, 4.4]]
        X2 = [[7.0, 5.0, 8.0], [7.5, 5.8, 8.4], [5.5, 5.8, 4.4]]
        kwargs = {'N0tV4l1D_p4raM': 3.14, 'w': np.arange(3)}
        args = [3.14] * 200
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, *args)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, **kwargs)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, kwarg=2.2, kwarg2=3.3)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, 1, 2, kwarg=2.2)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, 1, 2, kwarg=2.2)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, 1.1, 2.2, 3.3)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, 1.1, 2.2)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, 1.1)
        with pytest.raises(TypeError):
            cdist(X1, X2, _my_metric, 1.1, kwarg=2.2, kwarg2=3.3)
        assert_allclose(cdist(X1, X2, metric=_my_metric, arg=1.1, kwarg2=3.3), 5.4)

    def test_cdist_euclidean_random_unicode(self):
        eps = 1e-15
        X1 = eo['cdist-X1']
        X2 = eo['cdist-X2']
        Y1 = wcdist_no_const(X1, X2, 'euclidean')
        Y2 = wcdist_no_const(X1, X2, 'test_euclidean')
        assert_allclose(Y1, Y2, rtol=eps, verbose=verbose > 2)

    @pytest.mark.parametrize('p', [0.1, 0.25, 1.0, 1.23, 2.0, 3.8, 4.6, np.inf])
    def test_cdist_minkowski_random(self, p):
        eps = 1e-13
        X1 = eo['cdist-X1']
        X2 = eo['cdist-X2']
        Y1 = wcdist_no_const(X1, X2, 'minkowski', p=p)
        Y2 = wcdist_no_const(X1, X2, 'test_minkowski', p=p)
        assert_allclose(Y1, Y2, atol=0, rtol=eps, verbose=verbose > 2)

    def test_cdist_cosine_random(self):
        eps = 1e-14
        X1 = eo['cdist-X1']
        X2 = eo['cdist-X2']
        Y1 = wcdist(X1, X2, 'cosine')

        def norms(X):
            return np.linalg.norm(X, axis=1).reshape(-1, 1)
        Y2 = 1 - np.dot(X1 / norms(X1), (X2 / norms(X2)).T)
        assert_allclose(Y1, Y2, rtol=eps, verbose=verbose > 2)

    def test_cdist_mahalanobis(self):
        x1 = np.array([[2], [3]])
        x2 = np.array([[2], [5]])
        dist = cdist(x1, x2, metric='mahalanobis')
        assert_allclose(dist, [[0.0, np.sqrt(4.5)], [np.sqrt(0.5), np.sqrt(2)]])
        x1 = np.array([[0, 0], [-1, 0]])
        x2 = np.array([[0, 2], [1, 0], [0, -2]])
        dist = cdist(x1, x2, metric='mahalanobis')
        rt2 = np.sqrt(2)
        assert_allclose(dist, [[rt2, rt2, rt2], [2, 2 * rt2, 2]])
        with pytest.raises(ValueError):
            cdist([[0, 1]], [[2, 3]], metric='mahalanobis')

    def test_cdist_custom_notdouble(self):

        class myclass:
            pass

        def _my_metric(x, y):
            if not isinstance(x[0], myclass) or not isinstance(y[0], myclass):
                raise ValueError('Type has been changed')
            return 1.123
        data = np.array([[myclass()]], dtype=object)
        cdist_y = cdist(data, data, metric=_my_metric)
        right_y = 1.123
        assert_equal(cdist_y, right_y, verbose=verbose > 2)

    def _check_calling_conventions(self, X1, X2, metric, eps=1e-07, **kwargs):
        try:
            y1 = cdist(X1, X2, metric=metric, **kwargs)
            y2 = cdist(X1, X2, metric=eval(metric), **kwargs)
            y3 = cdist(X1, X2, metric='test_' + metric, **kwargs)
        except Exception as e:
            e_cls = e.__class__
            if verbose > 2:
                print(e_cls.__name__)
                print(e)
            with pytest.raises(e_cls):
                cdist(X1, X2, metric=metric, **kwargs)
            with pytest.raises(e_cls):
                cdist(X1, X2, metric=eval(metric), **kwargs)
            with pytest.raises(e_cls):
                cdist(X1, X2, metric='test_' + metric, **kwargs)
        else:
            assert_allclose(y1, y2, rtol=eps, verbose=verbose > 2)
            assert_allclose(y1, y3, rtol=eps, verbose=verbose > 2)

    def test_cdist_calling_conventions(self, metric):
        for eo_name in self.rnd_eo_names:
            X1 = eo[eo_name][::5, ::-2]
            X2 = eo[eo_name][1::5, ::2]
            if verbose > 2:
                print('testing: ', metric, ' with: ', eo_name)
            if metric in {'dice', 'yule', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'kulczynski1'} and 'bool' not in eo_name:
                continue
            self._check_calling_conventions(X1, X2, metric)
            if metric == 'seuclidean':
                X12 = np.vstack([X1, X2]).astype(np.float64)
                V = np.var(X12, axis=0, ddof=1)
                self._check_calling_conventions(X1, X2, metric, V=V)
            elif metric == 'mahalanobis':
                X12 = np.vstack([X1, X2]).astype(np.float64)
                V = np.atleast_2d(np.cov(X12.T))
                VI = np.array(np.linalg.inv(V).T)
                self._check_calling_conventions(X1, X2, metric, VI=VI)

    def test_cdist_dtype_equivalence(self, metric):
        eps = 1e-07
        tests = [(eo['random-bool-data'], self.valid_upcasts['bool']), (eo['random-uint-data'], self.valid_upcasts['uint']), (eo['random-int-data'], self.valid_upcasts['int']), (eo['random-float32-data'], self.valid_upcasts['float32'])]
        for test in tests:
            X1 = test[0][::5, ::-2]
            X2 = test[0][1::5, ::2]
            try:
                y1 = cdist(X1, X2, metric=metric)
            except Exception as e:
                e_cls = e.__class__
                if verbose > 2:
                    print(e_cls.__name__)
                    print(e)
                for new_type in test[1]:
                    X1new = new_type(X1)
                    X2new = new_type(X2)
                    with pytest.raises(e_cls):
                        cdist(X1new, X2new, metric=metric)
            else:
                for new_type in test[1]:
                    y2 = cdist(new_type(X1), new_type(X2), metric=metric)
                    assert_allclose(y1, y2, rtol=eps, verbose=verbose > 2)

    def test_cdist_out(self, metric):
        eps = 1e-15
        X1 = eo['cdist-X1']
        X2 = eo['cdist-X2']
        out_r, out_c = (X1.shape[0], X2.shape[0])
        kwargs = dict()
        if metric == 'minkowski':
            kwargs['p'] = 1.23
        out1 = np.empty((out_r, out_c), dtype=np.float64)
        Y1 = cdist(X1, X2, metric, **kwargs)
        Y2 = cdist(X1, X2, metric, out=out1, **kwargs)
        assert_allclose(Y1, Y2, rtol=eps, verbose=verbose > 2)
        assert_(Y2 is out1)
        out2 = np.empty((out_r - 1, out_c + 1), dtype=np.float64)
        with pytest.raises(ValueError):
            cdist(X1, X2, metric, out=out2, **kwargs)
        out3 = np.empty((2 * out_r, 2 * out_c), dtype=np.float64)[::2, ::2]
        out4 = np.empty((out_r, out_c), dtype=np.float64, order='F')
        with pytest.raises(ValueError):
            cdist(X1, X2, metric, out=out3, **kwargs)
        with pytest.raises(ValueError):
            cdist(X1, X2, metric, out=out4, **kwargs)
        out5 = np.empty((out_r, out_c), dtype=np.int64)
        with pytest.raises(ValueError):
            cdist(X1, X2, metric, out=out5, **kwargs)

    def test_striding(self, metric):
        eps = 1e-15
        X1 = eo['cdist-X1'][::2, ::2]
        X2 = eo['cdist-X2'][::2, ::2]
        X1_copy = X1.copy()
        X2_copy = X2.copy()
        assert_equal(X1, X1_copy)
        assert_equal(X2, X2_copy)
        assert_(not X1.flags.c_contiguous)
        assert_(not X2.flags.c_contiguous)
        assert_(X1_copy.flags.c_contiguous)
        assert_(X2_copy.flags.c_contiguous)
        kwargs = dict()
        if metric == 'minkowski':
            kwargs['p'] = 1.23
        Y1 = cdist(X1, X2, metric, **kwargs)
        Y2 = cdist(X1_copy, X2_copy, metric, **kwargs)
        assert_allclose(Y1, Y2, rtol=eps, verbose=verbose > 2)

    def test_cdist_refcount(self, metric):
        x1 = np.random.rand(10, 10)
        x2 = np.random.rand(10, 10)
        kwargs = dict()
        if metric == 'minkowski':
            kwargs['p'] = 1.23
        out = cdist(x1, x2, metric=metric, **kwargs)
        weak_refs = [weakref.ref(v) for v in (x1, x2, out)]
        del x1, x2, out
        if IS_PYPY:
            break_cycles()
        assert all((weak_ref() is None for weak_ref in weak_refs))