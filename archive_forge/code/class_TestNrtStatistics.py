import math
import os
import platform
import sys
import re
import numpy as np
from numba import njit
from numba.core import types
from numba.core.runtime import (
from numba.core.extending import intrinsic, include_path
from numba.core.typing import signature
from numba.core.imputils import impl_ret_untracked
from llvmlite import ir
import llvmlite.binding as llvm
from numba.core.unsafe.nrt import NRT_get_api
from numba.tests.support import (EnableNRTStatsMixin, TestCase, temp_directory,
from numba.core.registry import cpu_target
import unittest
class TestNrtStatistics(TestCase):

    def setUp(self):
        self.__stats_state = _nrt_python.memsys_stats_enabled()

    def tearDown(self):
        if self.__stats_state:
            _nrt_python.memsys_enable_stats()
        else:
            _nrt_python.memsys_disable_stats()

    def test_stats_env_var_explicit_on(self):
        src = 'if 1:\n        from numba import njit\n        import numpy as np\n        from numba.core.runtime import rtsys, _nrt_python\n        from numba.core.registry import cpu_target\n\n        @njit\n        def foo():\n            return np.arange(10)[0]\n\n        # initialize the NRT before use\n        rtsys.initialize(cpu_target.target_context)\n        assert _nrt_python.memsys_stats_enabled()\n        orig_stats = rtsys.get_allocation_stats()\n        foo()\n        new_stats = rtsys.get_allocation_stats()\n        total_alloc = new_stats.alloc - orig_stats.alloc\n        total_free = new_stats.free - orig_stats.free\n        total_mi_alloc = new_stats.mi_alloc - orig_stats.mi_alloc\n        total_mi_free = new_stats.mi_free - orig_stats.mi_free\n\n        expected = 1\n        assert total_alloc == expected\n        assert total_free == expected\n        assert total_mi_alloc == expected\n        assert total_mi_free == expected\n        '
        env = os.environ.copy()
        env['NUMBA_NRT_STATS'] = '1'
        run_in_subprocess(src, env=env)

    def check_env_var_off(self, env):
        src = 'if 1:\n        from numba import njit\n        import numpy as np\n        from numba.core.runtime import rtsys, _nrt_python\n\n        @njit\n        def foo():\n            return np.arange(10)[0]\n\n        assert _nrt_python.memsys_stats_enabled() == False\n        try:\n            rtsys.get_allocation_stats()\n        except RuntimeError as e:\n            assert "NRT stats are disabled." in str(e)\n        '
        run_in_subprocess(src, env=env)

    def test_stats_env_var_explicit_off(self):
        env = os.environ.copy()
        env['NUMBA_NRT_STATS'] = '0'
        self.check_env_var_off(env)

    def test_stats_env_var_default_off(self):
        env = os.environ.copy()
        env.pop('NUMBA_NRT_STATS', None)
        self.check_env_var_off(env)

    def test_stats_status_toggle(self):

        @njit
        def foo():
            tmp = np.ones(3)
            return np.arange(5 * tmp[0])
        _nrt_python.memsys_enable_stats()
        self.assertTrue(_nrt_python.memsys_stats_enabled())
        for i in range(2):
            stats_1 = rtsys.get_allocation_stats()
            _nrt_python.memsys_disable_stats()
            self.assertFalse(_nrt_python.memsys_stats_enabled())
            foo()
            _nrt_python.memsys_enable_stats()
            self.assertTrue(_nrt_python.memsys_stats_enabled())
            stats_2 = rtsys.get_allocation_stats()
            foo()
            stats_3 = rtsys.get_allocation_stats()
            self.assertEqual(stats_1, stats_2)
            self.assertLess(stats_2, stats_3)

    def test_rtsys_stats_query_raises_exception_when_disabled(self):
        _nrt_python.memsys_disable_stats()
        self.assertFalse(_nrt_python.memsys_stats_enabled())
        with self.assertRaises(RuntimeError) as raises:
            rtsys.get_allocation_stats()
        self.assertIn('NRT stats are disabled.', str(raises.exception))

    def test_nrt_explicit_stats_query_raises_exception_when_disabled(self):
        method_variations = ('alloc', 'free', 'mi_alloc', 'mi_free')
        for meth in method_variations:
            stats_func = getattr(_nrt_python, f'memsys_get_stats_{meth}')
            with self.subTest(stats_func=stats_func):
                _nrt_python.memsys_disable_stats()
                self.assertFalse(_nrt_python.memsys_stats_enabled())
                with self.assertRaises(RuntimeError) as raises:
                    stats_func()
                self.assertIn('NRT stats are disabled.', str(raises.exception))