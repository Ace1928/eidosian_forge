from __future__ import annotations
import warnings
from _pytest.recwarn import WarningsChecker
from pytest import warns
class NoWarningsChecker:

    def __init__(self):
        self.cw = warnings.catch_warnings(record=True)
        self.rec = []

    def __enter__(self):
        self.rec = self.cw.__enter__()

    def __exit__(self, type, value, traceback):
        if self.rec:
            warnings = [w.category.__name__ for w in self.rec]
            joined = '\\n'.join(warnings)
            raise AssertionError(f'Function is marked as not warning but the following warnings were found: \n{joined}')