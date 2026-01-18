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
class TestPlatformSpecificInfo(TestCase):

    def setUp(self):
        self.plat_spec_info = {'Linux': {str: (nsi._libc_version,)}, 'Windows': {str: (nsi._os_spec_version,)}, 'Darwin': {str: (nsi._os_spec_version,)}}
        self.os_name = platform.system()
        self.contents = self.plat_spec_info.get(self.os_name, {})
        self.info = nsi.get_os_spec_info(self.os_name)

    def test_has_all_data(self):
        keys = chain(*self.contents.values())
        for k in keys:
            with self.subTest(k=k):
                self.assertIn(k, self.info.keys())

    def test_content_type(self):
        for t, keys in self.contents.items():
            for k in keys:
                with self.subTest(k=k):
                    self.assertIsInstance(self.info[k], t)