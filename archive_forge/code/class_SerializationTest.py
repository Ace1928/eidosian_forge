import logging
import os
import pipes
import platform
import sys
import tempfile
import time
from unittest import mock
import testtools
from oslo_privsep import comm
from oslo_privsep import daemon
from oslo_privsep import priv_context
from oslo_privsep.tests import testctx
@testtools.skipIf(platform.system() != 'Linux', 'works only on Linux platform.')
class SerializationTest(testctx.TestContextTestCase):

    def test_basic_functionality(self):
        self.assertEqual(43, add1(42))

    def test_raises_standard(self):
        self.assertRaisesRegex(RuntimeError, "I can't let you do that Dave", fail)

    def test_raises_custom(self):
        exc = self.assertRaises(CustomError, fail, custom=True)
        self.assertEqual(exc.code, 42)
        self.assertEqual(exc.msg, 'omg!')