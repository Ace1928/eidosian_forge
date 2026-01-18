import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
class TestScalarRootFinders:
    xtol = 4 * np.finfo(float).eps
    rtol = 4 * np.finfo(float).eps

    def _run_one_test(self, tc, method, sig_args_keys=None, sig_kwargs_keys=None, **kwargs):
        method_args = []
        for k in sig_args_keys or []:
            if k not in tc:
                k = {'a': 'x0', 'b': 'x1', 'func': 'f'}.get(k, k)
            method_args.append(tc[k])
        method_kwargs = dict(**kwargs)
        method_kwargs.update({'full_output': True, 'disp': False})
        for k in sig_kwargs_keys or []:
            method_kwargs[k] = tc[k]
        root = tc.get('root')
        func_args = tc.get('args', ())
        try:
            r, rr = method(*method_args, args=func_args, **method_kwargs)
            return (root, rr, tc)
        except Exception:
            return (root, zeros.RootResults(nan, -1, -1, zeros._EVALUEERR, method), tc)

    def run_tests(self, tests, method, name, known_fail=None, **kwargs):
        """Run test-cases using the specified method and the supplied signature.

        Extract the arguments for the method call from the test case
        dictionary using the supplied keys for the method's signature."""
        sig = _getfullargspec(method)
        assert_(not sig.kwonlyargs)
        nDefaults = len(sig.defaults)
        nRequired = len(sig.args) - nDefaults
        sig_args_keys = sig.args[:nRequired]
        sig_kwargs_keys = []
        if name in ['secant', 'newton', 'halley']:
            if name in ['newton', 'halley']:
                sig_kwargs_keys.append('fprime')
                if name in ['halley']:
                    sig_kwargs_keys.append('fprime2')
            kwargs['tol'] = self.xtol
        else:
            kwargs['xtol'] = self.xtol
            kwargs['rtol'] = self.rtol
        results = [list(self._run_one_test(tc, method, sig_args_keys=sig_args_keys, sig_kwargs_keys=sig_kwargs_keys, **kwargs)) for tc in tests]
        known_fail = known_fail or []
        notcvgd = [elt for elt in results if not elt[1].converged]
        notcvgd = [elt for elt in notcvgd if elt[-1]['ID'] not in known_fail]
        notcvged_IDS = [elt[-1]['ID'] for elt in notcvgd]
        assert_equal([len(notcvged_IDS), notcvged_IDS], [0, []])
        tols = {'xtol': self.xtol, 'rtol': self.rtol}
        tols.update(**kwargs)
        rtol = tols['rtol']
        atol = tols.get('tol', tols['xtol'])
        cvgd = [elt for elt in results if elt[1].converged]
        approx = [elt[1].root for elt in cvgd]
        correct = [elt[0] for elt in cvgd]
        notclose = [[a] + elt for a, c, elt in zip(approx, correct, cvgd) if not isclose(a, c, rtol=rtol, atol=atol) and elt[-1]['ID'] not in known_fail]
        fvs = [tc['f'](aroot, *tc.get('args', tuple())) for aroot, c, fullout, tc in notclose]
        notclose = [[fv] + elt for fv, elt in zip(fvs, notclose) if fv != 0]
        assert_equal([notclose, len(notclose)], [[], 0])
        method_from_result = [result[1].method for result in results]
        expected_method = [name for _ in results]
        assert_equal(method_from_result, expected_method)

    def run_collection(self, collection, method, name, smoothness=None, known_fail=None, **kwargs):
        """Run a collection of tests using the specified method.

        The name is used to determine some optional arguments."""
        tests = get_tests(collection, smoothness=smoothness)
        self.run_tests(tests, method, name, known_fail=known_fail, **kwargs)