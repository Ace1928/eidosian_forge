import platform
import unittest
from unittest import skipUnless
from unittest.mock import NonCallableMock
from itertools import chain
from datetime import datetime
from contextlib import redirect_stdout
from io import StringIO
from numba.tests.support import TestCase
import numba.misc.numba_sysinfo as nsi
class TestSysInfo(TestCase):

    def setUp(self):
        super(TestSysInfo, self).setUp()
        self.info = nsi.get_sysinfo()
        self.safe_contents = {int: (nsi._cpu_count,), float: (nsi._runtime,), str: (nsi._machine, nsi._cpu_name, nsi._platform_name, nsi._os_name, nsi._os_version, nsi._python_comp, nsi._python_impl, nsi._python_version, nsi._llvm_version, nsi._numpy_version), bool: (nsi._cu_dev_init, nsi._svml_state, nsi._svml_loaded, nsi._svml_operational, nsi._llvm_svml_patched, nsi._tbb_thread, nsi._openmp_thread, nsi._wkq_thread, nsi._numpy_AVX512_SKX_detected), list: (nsi._errors, nsi._warnings), dict: (nsi._numba_env_vars,), datetime: (nsi._start, nsi._start_utc)}
        self.safe_keys = chain(*self.safe_contents.values())

    def tearDown(self):
        super(TestSysInfo, self).tearDown()
        del self.info

    def test_has_safe_keys(self):
        for k in self.safe_keys:
            with self.subTest(k=k):
                self.assertIn(k, self.info)

    def test_safe_content_type(self):
        for t, keys in self.safe_contents.items():
            for k in keys:
                with self.subTest(k=k):
                    self.assertIsInstance(self.info[k], t)

    def test_has_no_error(self):
        self.assertFalse(self.info[nsi._errors])

    def test_display_empty_info(self):
        output = StringIO()
        with redirect_stdout(output):
            res = nsi.display_sysinfo({})
        self.assertIsNone(res)
        output.close()