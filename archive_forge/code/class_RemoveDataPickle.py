from statsmodels.compat.pandas import assert_series_equal
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
class RemoveDataPickle:

    @classmethod
    def setup_class(cls):
        nobs = 1000
        np.random.seed(987689)
        x = np.random.randn(nobs, 3)
        x = sm.add_constant(x)
        cls.exog = x
        cls.xf = 0.25 * np.ones((2, 4))
        cls.predict_kwds = {}
        cls.reduction_factor = 0.1

    def test_remove_data_pickle(self):
        results = self.results
        xf = self.xf
        pred_kwds = self.predict_kwds
        pred1 = results.predict(xf, **pred_kwds)
        results.summary()
        results.summary2()
        res, orig_nbytes = check_pickle(results._results)
        results.remove_data()
        pred2 = results.predict(xf, **pred_kwds)
        if isinstance(pred1, pd.Series) and isinstance(pred2, pd.Series):
            assert_series_equal(pred1, pred2)
        elif isinstance(pred1, pd.DataFrame) and isinstance(pred2, pd.DataFrame):
            assert pred1.equals(pred2)
        else:
            np.testing.assert_equal(pred2, pred1)
        res, nbytes = check_pickle(results._results)
        self.res = res
        msg = 'pickle length not %d < %d' % (nbytes, orig_nbytes)
        assert nbytes < orig_nbytes * self.reduction_factor, msg
        pred3 = results.predict(xf, **pred_kwds)
        if isinstance(pred1, pd.Series) and isinstance(pred3, pd.Series):
            assert_series_equal(pred1, pred3)
        elif isinstance(pred1, pd.DataFrame) and isinstance(pred3, pd.DataFrame):
            assert pred1.equals(pred3)
        else:
            np.testing.assert_equal(pred3, pred1)

    def test_remove_data_docstring(self):
        assert self.results.remove_data.__doc__ is not None

    def test_pickle_wrapper(self):
        fh = BytesIO()
        self.results._results.save(fh)
        fh.seek(0, 0)
        res_unpickled = self.results._results.__class__.load(fh)
        assert type(res_unpickled) is type(self.results._results)
        fh.seek(0, 0)
        self.results.save(fh)
        fh.seek(0, 0)
        res_unpickled = self.results.__class__.load(fh)
        fh.close()
        assert type(res_unpickled) is type(self.results)
        before = sorted(self.results.__dict__.keys())
        after = sorted(res_unpickled.__dict__.keys())
        assert before == after, 'not equal {!r} and {!r}'.format(before, after)
        before = sorted(self.results._results.__dict__.keys())
        after = sorted(res_unpickled._results.__dict__.keys())
        assert before == after, 'not equal {!r} and {!r}'.format(before, after)
        before = sorted(self.results.model.__dict__.keys())
        after = sorted(res_unpickled.model.__dict__.keys())
        assert before == after, 'not equal {!r} and {!r}'.format(before, after)
        before = sorted(self.results._cache.keys())
        after = sorted(res_unpickled._cache.keys())
        assert before == after, 'not equal {!r} and {!r}'.format(before, after)