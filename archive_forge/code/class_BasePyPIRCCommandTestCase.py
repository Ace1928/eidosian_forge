import os
import unittest
from distutils.core import PyPIRCCommand
from distutils.core import Distribution
from distutils.log import set_threshold
from distutils.log import WARN
from distutils.tests import support
class BasePyPIRCCommandTestCase(support.TempdirManager, support.LoggingSilencer, support.EnvironGuard, unittest.TestCase):

    def setUp(self):
        """Patches the environment."""
        super(BasePyPIRCCommandTestCase, self).setUp()
        self.tmp_dir = self.mkdtemp()
        os.environ['HOME'] = self.tmp_dir
        os.environ['USERPROFILE'] = self.tmp_dir
        self.rc = os.path.join(self.tmp_dir, '.pypirc')
        self.dist = Distribution()

        class command(PyPIRCCommand):

            def __init__(self, dist):
                PyPIRCCommand.__init__(self, dist)

            def initialize_options(self):
                pass
            finalize_options = initialize_options
        self._cmd = command
        self.old_threshold = set_threshold(WARN)

    def tearDown(self):
        """Removes the patch."""
        set_threshold(self.old_threshold)
        super(BasePyPIRCCommandTestCase, self).tearDown()