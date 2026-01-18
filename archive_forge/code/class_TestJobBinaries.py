from osc_lib.tests import utils as osc_u
import testtools
from unittest import mock
from saharaclient.api import job_binaries as api_jb
from saharaclient.osc.v1 import job_binaries as osc_jb
from saharaclient.tests.unit.osc.v1 import test_job_binaries as tjb_v1
class TestJobBinaries(tjb_v1.TestJobBinaries):

    def setUp(self):
        super(TestJobBinaries, self).setUp()
        self.app.api_version['data_processing'] = '2'
        self.jb_mock = self.app.client_manager.data_processing.job_binaries
        self.jb_mock.reset_mock()