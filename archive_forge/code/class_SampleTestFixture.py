import doctest
import io
import sys
from textwrap import dedent
import unittest
from unittest import TestSuite
import testtools
from testtools import TestCase, run, skipUnless
from testtools.compat import _b
from testtools.helpers import try_import
from testtools.matchers import (
from testtools import TestCase
from fixtures import Fixture
from testresources import (
from testtools import TestCase
from testtools import TestCase, clone_test_with_new_id
class SampleTestFixture(fixtures.Fixture):
    """Creates testtools.runexample temporarily."""

    def __init__(self, broken=False):
        """Create a SampleTestFixture.

            :param broken: If True, the sample file will not be importable.
            """
        if not broken:
            init_contents = _b('from testtools import TestCase\n\nclass TestFoo(TestCase):\n    def test_bar(self):\n        pass\n    def test_quux(self):\n        pass\ndef test_suite():\n    from unittest import TestLoader\n    return TestLoader().loadTestsFromName(__name__)\n')
        else:
            init_contents = b'class not in\n'
        self.package = fixtures.PythonPackage('runexample', [('__init__.py', init_contents)])

    def setUp(self):
        super().setUp()
        self.useFixture(self.package)
        testtools.__path__.append(self.package.base)
        self.addCleanup(testtools.__path__.remove, self.package.base)
        self.addCleanup(sys.modules.pop, 'testtools.runexample', None)