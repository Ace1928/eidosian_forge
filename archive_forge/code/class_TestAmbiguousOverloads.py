from itertools import product, permutations
from collections import defaultdict
import unittest
from numba.core.base import OverloadSelector
from numba.core.registry import cpu_target
from numba.core.imputils import builtin_registry, RegistryLoader
from numba.core import types
from numba.core.errors import NumbaNotImplementedError, NumbaTypeError
class TestAmbiguousOverloads(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cpu_target.target_context.refresh()

    def create_overload_selector(self, kind):
        os = OverloadSelector()
        loader = RegistryLoader(builtin_registry)
        for impl, sig in loader.new_registrations(kind):
            os.append(impl, sig)
        return os

    def test_ambiguous_casts(self):
        os = self.create_overload_selector(kind='casts')
        all_types = set((t for sig, impl in os.versions for t in sig))
        for sig in permutations(all_types, r=2):
            try:
                os.find(sig)
            except NumbaNotImplementedError:
                pass

    def test_ambiguous_functions(self):
        loader = RegistryLoader(builtin_registry)
        selectors = defaultdict(OverloadSelector)
        for impl, fn, sig in loader.new_registrations('functions'):
            os = selectors[fn]
            os.append(impl, sig)
        for fn, os in selectors.items():
            all_types = set((t for sig, impl in os.versions for t in sig))
            for sig in product(all_types, all_types):
                try:
                    os.find(sig)
                except NumbaNotImplementedError:
                    pass