import os
import re
import sys
import numpy as np
import inspect
import sysconfig
class PytestTester:
    """
    Run tests for this namespace

    ``scipy.test()`` runs tests for all of SciPy, with the default settings.
    When used from a submodule (e.g., ``scipy.cluster.test()``, only the tests
    for that namespace are run.

    Parameters
    ----------
    label : {'fast', 'full'}, optional
        Whether to run only the fast tests, or also those marked as slow.
        Default is 'fast'.
    verbose : int, optional
        Test output verbosity. Default is 1.
    extra_argv : list, optional
        Arguments to pass through to Pytest.
    doctests : bool, optional
        Whether to run doctests or not. Default is False.
    coverage : bool, optional
        Whether to run tests with code coverage measurements enabled.
        Default is False.
    tests : list of str, optional
        List of module names to run tests for. By default, uses the module
        from which the ``test`` function is called.
    parallel : int, optional
        Run tests in parallel with pytest-xdist, if number given is larger than
        1. Default is 1.

    """

    def __init__(self, module_name):
        self.module_name = module_name

    def __call__(self, label='fast', verbose=1, extra_argv=None, doctests=False, coverage=False, tests=None, parallel=None):
        import pytest
        module = sys.modules[self.module_name]
        module_path = os.path.abspath(module.__path__[0])
        pytest_args = ['--showlocals', '--tb=short']
        if doctests:
            raise ValueError('Doctests not supported')
        if extra_argv:
            pytest_args += list(extra_argv)
        if verbose and int(verbose) > 1:
            pytest_args += ['-' + 'v' * (int(verbose) - 1)]
        if coverage:
            pytest_args += ['--cov=' + module_path]
        if label == 'fast':
            pytest_args += ['-m', 'not slow']
        elif label != 'full':
            pytest_args += ['-m', label]
        if tests is None:
            tests = [self.module_name]
        if parallel is not None and parallel > 1:
            if _pytest_has_xdist():
                pytest_args += ['-n', str(parallel)]
            else:
                import warnings
                warnings.warn('Could not run tests in parallel because pytest-xdist plugin is not available.', stacklevel=2)
        pytest_args += ['--pyargs'] + list(tests)
        try:
            code = pytest.main(pytest_args)
        except SystemExit as exc:
            code = exc.code
        return code == 0