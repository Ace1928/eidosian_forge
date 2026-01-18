from breezy.bzr.bzrdir import BzrDirFormat
from breezy.controldir import ControlDirFormat
from breezy.tests import (TestCaseWithTransport, default_transport,
from breezy.tests.per_controldir import make_scenarios
from breezy.transport import memory
class TestCaseWithBzrDir(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.controldir = None

    def get_bzrdir(self):
        if self.controldir is None:
            self.controldir = self.make_controldir(None)
        return self.controldir

    def get_default_format(self):
        return self.bzrdir_format