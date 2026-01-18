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
class TestSysInfoWithPsutil(TestCase):
    mem_total = 2 * 1024 ** 2
    mem_available = 1024 ** 2
    cpus_list = [1, 2]

    def setUp(self):
        super(TestSysInfoWithPsutil, self).setUp()
        self.psutil_orig_state = nsi._psutil_import
        nsi._psutil_import = True
        nsi.psutil = NonCallableMock()
        vm = nsi.psutil.virtual_memory.return_value
        vm.total = self.mem_total
        vm.available = self.mem_available
        if platform.system() in ('Linux', 'Windows'):
            proc = nsi.psutil.Process.return_value
            proc.cpu_affinity.return_value = self.cpus_list
        else:
            nsi.psutil.Process.return_value = None
        self.info = nsi.get_os_spec_info(platform.system())

    def tearDown(self):
        super(TestSysInfoWithPsutil, self).tearDown()
        nsi._psutil_import = self.psutil_orig_state

    def test_has_all_data(self):
        keys = (nsi._mem_total, nsi._mem_available)
        for k in keys:
            with self.subTest(k=k):
                self.assertIn(k, self.info.keys())
                self.assertIsInstance(self.info[k], int)

    def test_has_correct_values(self):
        self.assertEqual(self.info[nsi._mem_total], self.mem_total)
        self.assertEqual(self.info[nsi._mem_available], self.mem_available)

    @skipUnless(platform.system() in ('Linux', 'Windows'), 'CPUs allowed info only available on Linux and Windows')
    def test_cpus_list(self):
        self.assertEqual(self.info[nsi._cpus_allowed], len(self.cpus_list))
        self.assertEqual(self.info[nsi._cpus_list], ' '.join((str(n) for n in self.cpus_list)))