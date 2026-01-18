import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
class TestRandomDist:

    def setup_method(self):
        self.seed = 1234567890

    def test_rand(self):
        random.seed(self.seed)
        actual = random.rand(3, 2)
        desired = np.array([[0.61879477158568, 0.5916236277597466], [0.8886835890444966, 0.8916548001156082], [0.4575674820298663, 0.7781880808593471]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_rand_singleton(self):
        random.seed(self.seed)
        actual = random.rand()
        desired = 0.61879477158568
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_randn(self):
        random.seed(self.seed)
        actual = random.randn(3, 2)
        desired = np.array([[1.3401634577186312, 1.7375912277193608], [1.498988344300628, -0.2286433324536169], [2.031033998682787, 2.1703249460565526]])
        assert_array_almost_equal(actual, desired, decimal=15)
        random.seed(self.seed)
        actual = random.randn()
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_randint(self):
        random.seed(self.seed)
        actual = random.randint(-99, 99, size=(3, 2))
        desired = np.array([[31, 3], [-52, 41], [-48, -66]])
        assert_array_equal(actual, desired)

    def test_random_integers(self):
        random.seed(self.seed)
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = random.random_integers(-99, 99, size=(3, 2))
            assert_(len(w) == 1)
        desired = np.array([[31, 3], [-52, 41], [-48, -66]])
        assert_array_equal(actual, desired)
        random.seed(self.seed)
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = random.random_integers(198, size=(3, 2))
            assert_(len(w) == 1)
        assert_array_equal(actual, desired + 100)

    def test_tomaxint(self):
        random.seed(self.seed)
        rs = random.RandomState(self.seed)
        actual = rs.tomaxint(size=(3, 2))
        if np.iinfo(int).max == 2147483647:
            desired = np.array([[1328851649, 731237375], [1270502067, 320041495], [1908433478, 499156889]], dtype=np.int64)
        else:
            desired = np.array([[5707374374421908479, 5456764827585442327], [8196659375100692377, 8224063923314595285], [4220315081820346526, 7177518203184491332]], dtype=np.int64)
        assert_equal(actual, desired)
        rs.seed(self.seed)
        actual = rs.tomaxint()
        assert_equal(actual, desired[0, 0])

    def test_random_integers_max_int(self):
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            actual = random.random_integers(np.iinfo('l').max, np.iinfo('l').max)
            assert_(len(w) == 1)
        desired = np.iinfo('l').max
        assert_equal(actual, desired)
        with suppress_warnings() as sup:
            w = sup.record(DeprecationWarning)
            typer = np.dtype('l').type
            actual = random.random_integers(typer(np.iinfo('l').max), typer(np.iinfo('l').max))
            assert_(len(w) == 1)
        assert_equal(actual, desired)

    def test_random_integers_deprecated(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            assert_raises(DeprecationWarning, random.random_integers, np.iinfo('l').max)
            assert_raises(DeprecationWarning, random.random_integers, np.iinfo('l').max, np.iinfo('l').max)

    def test_random_sample(self):
        random.seed(self.seed)
        actual = random.random_sample((3, 2))
        desired = np.array([[0.61879477158568, 0.5916236277597466], [0.8886835890444966, 0.8916548001156082], [0.4575674820298663, 0.7781880808593471]])
        assert_array_almost_equal(actual, desired, decimal=15)
        random.seed(self.seed)
        actual = random.random_sample()
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_choice_uniform_replace(self):
        random.seed(self.seed)
        actual = random.choice(4, 4)
        desired = np.array([2, 3, 2, 3])
        assert_array_equal(actual, desired)

    def test_choice_nonuniform_replace(self):
        random.seed(self.seed)
        actual = random.choice(4, 4, p=[0.4, 0.4, 0.1, 0.1])
        desired = np.array([1, 1, 2, 2])
        assert_array_equal(actual, desired)

    def test_choice_uniform_noreplace(self):
        random.seed(self.seed)
        actual = random.choice(4, 3, replace=False)
        desired = np.array([0, 1, 3])
        assert_array_equal(actual, desired)

    def test_choice_nonuniform_noreplace(self):
        random.seed(self.seed)
        actual = random.choice(4, 3, replace=False, p=[0.1, 0.3, 0.5, 0.1])
        desired = np.array([2, 3, 1])
        assert_array_equal(actual, desired)

    def test_choice_noninteger(self):
        random.seed(self.seed)
        actual = random.choice(['a', 'b', 'c', 'd'], 4)
        desired = np.array(['c', 'd', 'c', 'd'])
        assert_array_equal(actual, desired)

    def test_choice_exceptions(self):
        sample = random.choice
        assert_raises(ValueError, sample, -1, 3)
        assert_raises(ValueError, sample, 3.0, 3)
        assert_raises(ValueError, sample, [[1, 2], [3, 4]], 3)
        assert_raises(ValueError, sample, [], 3)
        assert_raises(ValueError, sample, [1, 2, 3, 4], 3, p=[[0.25, 0.25], [0.25, 0.25]])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4, 0.2])
        assert_raises(ValueError, sample, [1, 2], 3, p=[1.1, -0.1])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4])
        assert_raises(ValueError, sample, [1, 2, 3], 4, replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], -2, replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1,), replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1, 1), replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], 2, replace=False, p=[1, 0, 0])

    def test_choice_return_shape(self):
        p = [0.1, 0.9]
        assert_(np.isscalar(random.choice(2, replace=True)))
        assert_(np.isscalar(random.choice(2, replace=False)))
        assert_(np.isscalar(random.choice(2, replace=True, p=p)))
        assert_(np.isscalar(random.choice(2, replace=False, p=p)))
        assert_(np.isscalar(random.choice([1, 2], replace=True)))
        assert_(random.choice([None], replace=True) is None)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, replace=True) is a)
        s = tuple()
        assert_(not np.isscalar(random.choice(2, s, replace=True)))
        assert_(not np.isscalar(random.choice(2, s, replace=False)))
        assert_(not np.isscalar(random.choice(2, s, replace=True, p=p)))
        assert_(not np.isscalar(random.choice(2, s, replace=False, p=p)))
        assert_(not np.isscalar(random.choice([1, 2], s, replace=True)))
        assert_(random.choice([None], s, replace=True).ndim == 0)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, s, replace=True).item() is a)
        s = (2, 3)
        p = [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]
        assert_equal(random.choice(6, s, replace=True).shape, s)
        assert_equal(random.choice(6, s, replace=False).shape, s)
        assert_equal(random.choice(6, s, replace=True, p=p).shape, s)
        assert_equal(random.choice(6, s, replace=False, p=p).shape, s)
        assert_equal(random.choice(np.arange(6), s, replace=True).shape, s)
        assert_equal(random.randint(0, 0, size=(3, 0, 4)).shape, (3, 0, 4))
        assert_equal(random.randint(0, -10, size=0).shape, (0,))
        assert_equal(random.randint(10, 10, size=0).shape, (0,))
        assert_equal(random.choice(0, size=0).shape, (0,))
        assert_equal(random.choice([], size=(0,)).shape, (0,))
        assert_equal(random.choice(['a', 'b'], size=(3, 0, 4)).shape, (3, 0, 4))
        assert_raises(ValueError, random.choice, [], 10)

    def test_choice_nan_probabilities(self):
        a = np.array([42, 1, 2])
        p = [None, None, None]
        assert_raises(ValueError, random.choice, a, p=p)

    def test_choice_p_non_contiguous(self):
        p = np.ones(10) / 5
        p[1::2] = 3.0
        random.seed(self.seed)
        non_contig = random.choice(5, 3, p=p[::2])
        random.seed(self.seed)
        contig = random.choice(5, 3, p=np.ascontiguousarray(p[::2]))
        assert_array_equal(non_contig, contig)

    def test_bytes(self):
        random.seed(self.seed)
        actual = random.bytes(10)
        desired = b'\x82Ui\x9e\xff\x97+Wf\xa5'
        assert_equal(actual, desired)

    def test_shuffle(self):
        for conv in [lambda x: np.array([]), lambda x: x, lambda x: np.asarray(x).astype(np.int8), lambda x: np.asarray(x).astype(np.float32), lambda x: np.asarray(x).astype(np.complex64), lambda x: np.asarray(x).astype(object), lambda x: [(i, i) for i in x], lambda x: np.asarray([[i, i] for i in x]), lambda x: np.vstack([x, x]).T, lambda x: np.asarray([(i, i) for i in x], [('a', int), ('b', int)]).view(np.recarray), lambda x: np.asarray([(i, i) for i in x], [('a', object, (1,)), ('b', np.int32, (1,))])]:
            random.seed(self.seed)
            alist = conv([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
            random.shuffle(alist)
            actual = alist
            desired = conv([0, 1, 9, 6, 2, 4, 5, 8, 7, 3])
            assert_array_equal(actual, desired)

    def test_shuffle_masked(self):
        a = np.ma.masked_values(np.reshape(range(20), (5, 4)) % 3 - 1, -1)
        b = np.ma.masked_values(np.arange(20) % 3 - 1, -1)
        a_orig = a.copy()
        b_orig = b.copy()
        for i in range(50):
            random.shuffle(a)
            assert_equal(sorted(a.data[~a.mask]), sorted(a_orig.data[~a_orig.mask]))
            random.shuffle(b)
            assert_equal(sorted(b.data[~b.mask]), sorted(b_orig.data[~b_orig.mask]))

        def test_shuffle_invalid_objects(self):
            x = np.array(3)
            assert_raises(TypeError, random.shuffle, x)

    def test_permutation(self):
        random.seed(self.seed)
        alist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        actual = random.permutation(alist)
        desired = [0, 1, 9, 6, 2, 4, 5, 8, 7, 3]
        assert_array_equal(actual, desired)
        random.seed(self.seed)
        arr_2d = np.atleast_2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]).T
        actual = random.permutation(arr_2d)
        assert_array_equal(actual, np.atleast_2d(desired).T)
        random.seed(self.seed)
        bad_x_str = 'abcd'
        assert_raises(IndexError, random.permutation, bad_x_str)
        random.seed(self.seed)
        bad_x_float = 1.2
        assert_raises(IndexError, random.permutation, bad_x_float)
        integer_val = 10
        desired = [9, 0, 8, 5, 1, 3, 4, 7, 6, 2]
        random.seed(self.seed)
        actual = random.permutation(integer_val)
        assert_array_equal(actual, desired)

    def test_beta(self):
        random.seed(self.seed)
        actual = random.beta(0.1, 0.9, size=(3, 2))
        desired = np.array([[0.014534185051374606, 0.0005312976156628681], [1.8536661905843232e-06, 0.004192145168001106], [0.0001584051551084981, 0.00012625289194939765]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_binomial(self):
        random.seed(self.seed)
        actual = random.binomial(100.123, 0.456, size=(3, 2))
        desired = np.array([[37, 43], [42, 48], [46, 45]])
        assert_array_equal(actual, desired)
        random.seed(self.seed)
        actual = random.binomial(100.123, 0.456)
        desired = 37
        assert_array_equal(actual, desired)

    def test_chisquare(self):
        random.seed(self.seed)
        actual = random.chisquare(50, size=(3, 2))
        desired = np.array([[63.878581755010906, 68.6840774891137], [65.77116116901506, 47.096867624389745], [72.38284031996952, 74.18408615260374]])
        assert_array_almost_equal(actual, desired, decimal=13)

    def test_dirichlet(self):
        random.seed(self.seed)
        alpha = np.array([51.72840233779265, 39.74494232180944])
        actual = random.dirichlet(alpha, size=(3, 2))
        desired = np.array([[[0.5453944457361156, 0.4546055542638844], [0.6234581682203941, 0.376541831779606]], [[0.5520600008578578, 0.44793999914214233], [0.589640233051543, 0.4103597669484569]], [[0.5926690928064783, 0.4073309071935218], [0.5697443174397521, 0.430255682560248]]])
        assert_array_almost_equal(actual, desired, decimal=15)
        bad_alpha = np.array([0.54, -1e-16])
        assert_raises(ValueError, random.dirichlet, bad_alpha)
        random.seed(self.seed)
        alpha = np.array([51.72840233779265, 39.74494232180944])
        actual = random.dirichlet(alpha)
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_dirichlet_size(self):
        p = np.array([51.72840233779265, 39.74494232180944])
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, [2, 2]).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, (2, 2)).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, np.array((2, 2))).shape, (2, 2, 2))
        assert_raises(TypeError, random.dirichlet, p, float(1))

    def test_dirichlet_bad_alpha(self):
        alpha = np.array([0.54, -1e-16])
        assert_raises(ValueError, random.dirichlet, alpha)

    def test_dirichlet_alpha_non_contiguous(self):
        a = np.array([51.72840233779265, -1.0, 39.74494232180944])
        alpha = a[::2]
        random.seed(self.seed)
        non_contig = random.dirichlet(alpha, size=(3, 2))
        random.seed(self.seed)
        contig = random.dirichlet(np.ascontiguousarray(alpha), size=(3, 2))
        assert_array_almost_equal(non_contig, contig)

    def test_exponential(self):
        random.seed(self.seed)
        actual = random.exponential(1.1234, size=(3, 2))
        desired = np.array([[1.0834264977501162, 1.0060788992455731], [2.466288300852167, 2.496681068099239], [0.6871743346136344, 1.6917566699357598]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_exponential_0(self):
        assert_equal(random.exponential(scale=0), 0)
        assert_raises(ValueError, random.exponential, scale=-0.0)

    def test_f(self):
        random.seed(self.seed)
        actual = random.f(12, 77, size=(3, 2))
        desired = np.array([[1.2197539441857588, 1.7513575979155978], [1.4480311501714649, 1.2210895948039626], [1.0217697575774063, 1.3443182762330042]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_gamma(self):
        random.seed(self.seed)
        actual = random.gamma(5, 3, size=(3, 2))
        desired = np.array([[24.605091886492872, 28.549935632072106], [26.134761102040642, 12.56988482927716], [31.718632757899606, 33.30143302795922]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_gamma_0(self):
        assert_equal(random.gamma(shape=0, scale=0), 0)
        assert_raises(ValueError, random.gamma, shape=-0.0, scale=-0.0)

    def test_geometric(self):
        random.seed(self.seed)
        actual = random.geometric(0.123456789, size=(3, 2))
        desired = np.array([[8, 7], [17, 17], [5, 12]])
        assert_array_equal(actual, desired)

    def test_geometric_exceptions(self):
        assert_raises(ValueError, random.geometric, 1.1)
        assert_raises(ValueError, random.geometric, [1.1] * 10)
        assert_raises(ValueError, random.geometric, -0.1)
        assert_raises(ValueError, random.geometric, [-0.1] * 10)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)
            assert_raises(ValueError, random.geometric, np.nan)
            assert_raises(ValueError, random.geometric, [np.nan] * 10)

    def test_gumbel(self):
        random.seed(self.seed)
        actual = random.gumbel(loc=0.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[0.19591898743416816, 0.34405539668096674], [-1.4492522252274278, -1.4737481629844686], [1.1065109047880342, -0.6953584862623617]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_gumbel_0(self):
        assert_equal(random.gumbel(scale=0), 0)
        assert_raises(ValueError, random.gumbel, scale=-0.0)

    def test_hypergeometric(self):
        random.seed(self.seed)
        actual = random.hypergeometric(10.1, 5.5, 14, size=(3, 2))
        desired = np.array([[10, 10], [10, 10], [9, 9]])
        assert_array_equal(actual, desired)
        actual = random.hypergeometric(5, 0, 3, size=4)
        desired = np.array([3, 3, 3, 3])
        assert_array_equal(actual, desired)
        actual = random.hypergeometric(15, 0, 12, size=4)
        desired = np.array([12, 12, 12, 12])
        assert_array_equal(actual, desired)
        actual = random.hypergeometric(0, 5, 3, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)
        actual = random.hypergeometric(0, 15, 12, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

    def test_laplace(self):
        random.seed(self.seed)
        actual = random.laplace(loc=0.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[0.6659972111276016, 0.5282945255222194], [3.1279195951440713, 3.18202813572992], [-0.05391065675859356, 1.7490133624283732]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_laplace_0(self):
        assert_equal(random.laplace(scale=0), 0)
        assert_raises(ValueError, random.laplace, scale=-0.0)

    def test_logistic(self):
        random.seed(self.seed)
        actual = random.logistic(loc=0.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[1.0923283530501144, 0.8648196662399954], [4.278185906949502, 4.338970063469297], [-0.21682183359214885, 2.6337336538606033]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_lognormal(self):
        random.seed(self.seed)
        actual = random.lognormal(mean=0.123456789, sigma=2.0, size=(3, 2))
        desired = np.array([[16.506986316888838, 36.54846706092655], [22.678865999812817, 0.7161756105899577], [65.72798501792724, 86.84341601437161]])
        assert_array_almost_equal(actual, desired, decimal=13)

    def test_lognormal_0(self):
        assert_equal(random.lognormal(sigma=0), 1)
        assert_raises(ValueError, random.lognormal, sigma=-0.0)

    def test_logseries(self):
        random.seed(self.seed)
        actual = random.logseries(p=0.923456789, size=(3, 2))
        desired = np.array([[2, 2], [6, 17], [3, 6]])
        assert_array_equal(actual, desired)

    def test_logseries_zero(self):
        assert random.logseries(0) == 1

    @pytest.mark.parametrize('value', [np.nextafter(0.0, -1), 1.0, np.nan, 5.0])
    def test_logseries_exceptions(self, value):
        with np.errstate(invalid='ignore'):
            with pytest.raises(ValueError):
                random.logseries(value)
            with pytest.raises(ValueError):
                random.logseries(np.array([value] * 10))
            with pytest.raises(ValueError):
                random.logseries(np.array([value] * 10)[::2])

    def test_multinomial(self):
        random.seed(self.seed)
        actual = random.multinomial(20, [1 / 6.0] * 6, size=(3, 2))
        desired = np.array([[[4, 3, 5, 4, 2, 2], [5, 2, 8, 2, 2, 1]], [[3, 4, 3, 6, 0, 4], [2, 1, 4, 3, 6, 4]], [[4, 4, 2, 5, 2, 3], [4, 3, 4, 2, 3, 4]]])
        assert_array_equal(actual, desired)

    def test_multivariate_normal(self):
        random.seed(self.seed)
        mean = (0.123456789, 10)
        cov = [[1, 0], [0, 1]]
        size = (3, 2)
        actual = random.multivariate_normal(mean, cov, size)
        desired = np.array([[[1.463620246718631, 11.73759122771936], [1.622445133300628, 9.771356667546383]], [[2.154490787682787, 12.170324946056553], [1.719909438201865, 9.230548443648306]], [[0.689515026297799, 9.880729819607714], [-0.023054015651998, 9.20109662354288]]])
        assert_array_almost_equal(actual, desired, decimal=15)
        actual = random.multivariate_normal(mean, cov)
        desired = np.array([0.895289569463708, 9.17180864067987])
        assert_array_almost_equal(actual, desired, decimal=15)
        mean = [0, 0]
        cov = [[1, 2], [2, 1]]
        assert_warns(RuntimeWarning, random.multivariate_normal, mean, cov)
        assert_no_warnings(random.multivariate_normal, mean, cov, check_valid='ignore')
        assert_raises(ValueError, random.multivariate_normal, mean, cov, check_valid='raise')
        cov = np.array([[1, 0.1], [0.1, 1]], dtype=np.float32)
        with suppress_warnings() as sup:
            random.multivariate_normal(mean, cov)
            w = sup.record(RuntimeWarning)
            assert len(w) == 0
        mu = np.zeros(2)
        cov = np.eye(2)
        assert_raises(ValueError, random.multivariate_normal, mean, cov, check_valid='other')
        assert_raises(ValueError, random.multivariate_normal, np.zeros((2, 1, 1)), cov)
        assert_raises(ValueError, random.multivariate_normal, mu, np.empty((3, 2)))
        assert_raises(ValueError, random.multivariate_normal, mu, np.eye(3))

    def test_negative_binomial(self):
        random.seed(self.seed)
        actual = random.negative_binomial(n=100, p=0.12345, size=(3, 2))
        desired = np.array([[848, 841], [892, 611], [779, 647]])
        assert_array_equal(actual, desired)

    def test_negative_binomial_exceptions(self):
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)
            assert_raises(ValueError, random.negative_binomial, 100, np.nan)
            assert_raises(ValueError, random.negative_binomial, 100, [np.nan] * 10)

    def test_noncentral_chisquare(self):
        random.seed(self.seed)
        actual = random.noncentral_chisquare(df=5, nonc=5, size=(3, 2))
        desired = np.array([[23.919053544985175, 13.353246927338263], [31.224526613297364, 16.600473994661773], [5.034615982627246, 17.949730890235195]])
        assert_array_almost_equal(actual, desired, decimal=14)
        actual = random.noncentral_chisquare(df=0.5, nonc=0.2, size=(3, 2))
        desired = np.array([[1.4714537782851667, 0.1505289926801266], [0.00943803056963588, 1.0264725161566617], [0.332334982684171, 0.15451287602753125]])
        assert_array_almost_equal(actual, desired, decimal=14)
        random.seed(self.seed)
        actual = random.noncentral_chisquare(df=5, nonc=0, size=(3, 2))
        desired = np.array([[9.597154162763948, 11.72548445029608], [10.413711048138335, 3.694475922923986], [13.484222138963087, 14.377255424602957]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_noncentral_f(self):
        random.seed(self.seed)
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=1, size=(3, 2))
        desired = np.array([[1.4059809967492667, 0.3420797317928576], [3.5771506926577255, 7.926326625778298], [0.4374159946354416, 1.177420875242832]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_noncentral_f_nan(self):
        random.seed(self.seed)
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=np.nan)
        assert np.isnan(actual)

    def test_normal(self):
        random.seed(self.seed)
        actual = random.normal(loc=0.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[2.8037837044372624, 3.5986392444387216], [3.121433477601256, -0.3338298759072338], [4.185524786365574, 4.464106681113105]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_normal_0(self):
        assert_equal(random.normal(scale=0), 0)
        assert_raises(ValueError, random.normal, scale=-0.0)

    def test_pareto(self):
        random.seed(self.seed)
        actual = random.pareto(a=0.123456789, size=(3, 2))
        desired = np.array([[2468.5246043903485, 1412.8688081051835], [52828779.70294852, 65772098.10473288], [140.84032335039151, 198390.2551352517]])
        np.testing.assert_array_almost_equal_nulp(actual, desired, nulp=30)

    def test_poisson(self):
        random.seed(self.seed)
        actual = random.poisson(lam=0.123456789, size=(3, 2))
        desired = np.array([[0, 0], [1, 0], [0, 0]])
        assert_array_equal(actual, desired)

    def test_poisson_exceptions(self):
        lambig = np.iinfo('l').max
        lamneg = -1
        assert_raises(ValueError, random.poisson, lamneg)
        assert_raises(ValueError, random.poisson, [lamneg] * 10)
        assert_raises(ValueError, random.poisson, lambig)
        assert_raises(ValueError, random.poisson, [lambig] * 10)
        with suppress_warnings() as sup:
            sup.record(RuntimeWarning)
            assert_raises(ValueError, random.poisson, np.nan)
            assert_raises(ValueError, random.poisson, [np.nan] * 10)

    def test_power(self):
        random.seed(self.seed)
        actual = random.power(a=0.123456789, size=(3, 2))
        desired = np.array([[0.02048932883240791, 0.01424192241128213], [0.384460737485353, 0.39499689943484395], [0.00177699707563439, 0.13115505880863756]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_rayleigh(self):
        random.seed(self.seed)
        actual = random.rayleigh(scale=10, size=(3, 2))
        desired = np.array([[13.88824964942484, 13.383318339044731], [20.95413364294492, 21.082850158007126], [11.060665370068543, 17.35468505778271]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_rayleigh_0(self):
        assert_equal(random.rayleigh(scale=0), 0)
        assert_raises(ValueError, random.rayleigh, scale=-0.0)

    def test_standard_cauchy(self):
        random.seed(self.seed)
        actual = random.standard_cauchy(size=(3, 2))
        desired = np.array([[0.7712766019644534, -6.556011619559106], [0.9358202339115831, -2.0747929301375945], [-4.746016442970119, 0.18338989290760804]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_exponential(self):
        random.seed(self.seed)
        actual = random.standard_exponential(size=(3, 2))
        desired = np.array([[0.964417391623746, 0.8955660488210551], [2.195378583631981, 2.2224328539249054], [0.6116915921431676, 1.505925467274132]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_gamma(self):
        random.seed(self.seed)
        actual = random.standard_gamma(shape=3, size=(3, 2))
        desired = np.array([[5.508415313184551, 6.629534703019031], [5.939884849437792, 2.31044849402134], [7.548386142313171, 8.012756093271868]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_standard_gamma_0(self):
        assert_equal(random.standard_gamma(shape=0), 0)
        assert_raises(ValueError, random.standard_gamma, shape=-0.0)

    def test_standard_normal(self):
        random.seed(self.seed)
        actual = random.standard_normal(size=(3, 2))
        desired = np.array([[1.3401634577186312, 1.7375912277193608], [1.498988344300628, -0.2286433324536169], [2.031033998682787, 2.1703249460565526]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_randn_singleton(self):
        random.seed(self.seed)
        actual = random.randn()
        desired = np.array(1.3401634577186312)
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_t(self):
        random.seed(self.seed)
        actual = random.standard_t(df=10, size=(3, 2))
        desired = np.array([[0.9714061186265996, -0.08830486548450577], [1.3631114368950532, -0.5531746390986707], [-0.18473749069684214, 0.6118153734175532]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_triangular(self):
        random.seed(self.seed)
        actual = random.triangular(left=5.12, mode=10.23, right=20.34, size=(3, 2))
        desired = np.array([[12.681171789492158, 12.412920614919315], [16.201313773351583, 16.256921387476005], [11.204006909118203, 14.497814483582992]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_uniform(self):
        random.seed(self.seed)
        actual = random.uniform(low=1.23, high=10.54, size=(3, 2))
        desired = np.array([[6.99097932346268, 6.73801597444324], [9.503644214004263, 9.53130618907631], [5.489953257698055, 8.474931032800521]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_uniform_range_bounds(self):
        fmin = np.finfo('float').min
        fmax = np.finfo('float').max
        func = random.uniform
        assert_raises(OverflowError, func, -np.inf, 0)
        assert_raises(OverflowError, func, 0, np.inf)
        assert_raises(OverflowError, func, fmin, fmax)
        assert_raises(OverflowError, func, [-np.inf], [0])
        assert_raises(OverflowError, func, [0], [np.inf])
        random.uniform(low=np.nextafter(fmin, 1), high=fmax / 1e+17)

    def test_scalar_exception_propagation(self):

        class ThrowingFloat(np.ndarray):

            def __float__(self):
                raise TypeError
        throwing_float = np.array(1.0).view(ThrowingFloat)
        assert_raises(TypeError, random.uniform, throwing_float, throwing_float)

        class ThrowingInteger(np.ndarray):

            def __int__(self):
                raise TypeError
        throwing_int = np.array(1).view(ThrowingInteger)
        assert_raises(TypeError, random.hypergeometric, throwing_int, 1, 1)

    def test_vonmises(self):
        random.seed(self.seed)
        actual = random.vonmises(mu=1.23, kappa=1.54, size=(3, 2))
        desired = np.array([[2.2856757267390204, 2.8916383844228504], [0.38198375564286025, 2.5763802311389075], [1.1915377158835305, 1.8350984968182535]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_vonmises_small(self):
        random.seed(self.seed)
        r = random.vonmises(mu=0.0, kappa=1.1e-08, size=10 ** 6)
        assert_(np.isfinite(r).all())

    def test_vonmises_large(self):
        random.seed(self.seed)
        actual = random.vonmises(mu=0.0, kappa=10000000.0, size=3)
        desired = np.array([0.0004634253748521111, 0.0003558873596114509, -0.0002337119622577433])
        assert_array_almost_equal(actual, desired, decimal=8)

    def test_vonmises_nan(self):
        random.seed(self.seed)
        r = random.vonmises(mu=0.0, kappa=np.nan)
        assert_(np.isnan(r))

    def test_wald(self):
        random.seed(self.seed)
        actual = random.wald(mean=1.23, scale=1.54, size=(3, 2))
        desired = np.array([[3.8293526571589, 5.131252491842855], [0.35045403618358717, 1.5083239687200354], [0.24124319895843183, 0.22031101461955038]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_weibull(self):
        random.seed(self.seed)
        actual = random.weibull(a=1.23, size=(3, 2))
        desired = np.array([[0.9709734264876673, 0.9142289644356552], [1.8951777003496293, 1.9141435796047956], [0.6705778375239099, 1.394940466350668]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_weibull_0(self):
        random.seed(self.seed)
        assert_equal(random.weibull(a=0, size=12), np.zeros(12))
        assert_raises(ValueError, random.weibull, a=-0.0)

    def test_zipf(self):
        random.seed(self.seed)
        actual = random.zipf(a=1.23, size=(3, 2))
        desired = np.array([[66, 29], [1, 1], [3, 13]])
        assert_array_equal(actual, desired)