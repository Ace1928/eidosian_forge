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
class TestSysInfoWithoutPsutil(TestCase):

    def setUp(self):
        super(TestSysInfoWithoutPsutil, self).setUp()
        self.psutil_orig_state = nsi._psutil_import
        nsi._psutil_import = False
        self.info = nsi.get_os_spec_info(platform.system())

    def tearDown(self):
        super(TestSysInfoWithoutPsutil, self).tearDown()
        nsi._psutil_import = self.psutil_orig_state

    def test_has_all_data(self):
        keys = (nsi._mem_total, nsi._mem_available)
        for k in keys:
            with self.subTest(k=k):
                self.assertIn(k, self.info.keys())
                self.assertIsInstance(self.info[k], int)