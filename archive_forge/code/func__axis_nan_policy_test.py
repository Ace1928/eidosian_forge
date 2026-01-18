from itertools import product, combinations_with_replacement, permutations
import re
import pickle
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy.stats import norm  # type: ignore[attr-defined]
from scipy.stats._axis_nan_policy import _masked_arrays_2_sentinel_arrays
from scipy._lib._util import AxisError
def _axis_nan_policy_test(hypotest, args, kwds, n_samples, n_outputs, paired, unpacker, nan_policy, axis, data_generator):
    if not unpacker:

        def unpacker(res):
            return res
    rng = np.random.default_rng(0)
    n_repetitions = 3
    data_gen_kwds = {'n_samples': n_samples, 'n_repetitions': n_repetitions, 'axis': axis, 'rng': rng, 'paired': paired}
    if data_generator == 'mixed':
        inherent_size = 6
        data = _mixed_data_generator(**data_gen_kwds)
    elif data_generator == 'all_nans':
        inherent_size = 2
        data_gen_kwds['all_nans'] = True
        data = _homogeneous_data_generator(**data_gen_kwds)
    elif data_generator == 'all_finite':
        inherent_size = 2
        data_gen_kwds['all_nans'] = False
        data = _homogeneous_data_generator(**data_gen_kwds)
    output_shape = [n_repetitions] + [inherent_size] * n_samples
    data_b = [np.moveaxis(sample, axis, -1) for sample in data]
    data_b = [np.broadcast_to(sample, output_shape + [sample.shape[-1]]) for sample in data_b]
    statistics = np.zeros(output_shape)
    pvalues = np.zeros(output_shape)
    for i, _ in np.ndenumerate(statistics):
        data1d = [sample[i] for sample in data_b]
        with np.errstate(divide='ignore', invalid='ignore'):
            try:
                res1d = nan_policy_1d(hypotest, data1d, unpacker, *args, n_outputs=n_outputs, nan_policy=nan_policy, paired=paired, _no_deco=True, **kwds)
                res1db = unpacker(hypotest(*data1d, *args, nan_policy=nan_policy, **kwds))
                assert_equal(res1db[0], res1d[0])
                if len(res1db) == 2:
                    assert_equal(res1db[1], res1d[1])
            except (RuntimeWarning, UserWarning, ValueError, ZeroDivisionError) as e:
                with pytest.raises(type(e), match=re.escape(str(e))):
                    nan_policy_1d(hypotest, data1d, unpacker, *args, n_outputs=n_outputs, nan_policy=nan_policy, paired=paired, _no_deco=True, **kwds)
                with pytest.raises(type(e), match=re.escape(str(e))):
                    hypotest(*data1d, *args, nan_policy=nan_policy, **kwds)
                if any([str(e).startswith(message) for message in too_small_messages]):
                    res1d = np.full(n_outputs, np.nan)
                elif any([str(e).startswith(message) for message in inaccuracy_messages]):
                    with suppress_warnings() as sup:
                        sup.filter(RuntimeWarning)
                        sup.filter(UserWarning)
                        res1d = nan_policy_1d(hypotest, data1d, unpacker, *args, n_outputs=n_outputs, nan_policy=nan_policy, paired=paired, _no_deco=True, **kwds)
                else:
                    raise e
        statistics[i] = res1d[0]
        if len(res1d) == 2:
            pvalues[i] = res1d[1]
    if nan_policy == 'raise' and (not data_generator == 'all_finite'):
        message = 'The input contains nan values'
        with pytest.raises(ValueError, match=message):
            hypotest(*data, *args, axis=axis, nan_policy=nan_policy, **kwds)
    else:
        with suppress_warnings() as sup, np.errstate(divide='ignore', invalid='ignore'):
            sup.filter(RuntimeWarning, 'Precision loss occurred in moment')
            sup.filter(UserWarning, 'Sample size too small for normal approximation.')
            res = unpacker(hypotest(*data, *args, axis=axis, nan_policy=nan_policy, **kwds))
        assert_allclose(res[0], statistics, rtol=1e-15)
        assert_equal(res[0].dtype, statistics.dtype)
        if len(res) == 2:
            assert_allclose(res[1], pvalues, rtol=1e-15)
            assert_equal(res[1].dtype, pvalues.dtype)