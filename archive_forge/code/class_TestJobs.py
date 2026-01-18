from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api.v2 import jobs as api_j
from saharaclient.osc.v2 import jobs as osc_j
from saharaclient.tests.unit.osc.v1 import test_jobs as tj_v1
class TestJobs(tj_v1.TestJobs):

    def setUp(self):
        super(TestJobs, self).setUp()
        self.app.api_version['data_processing'] = '2'
        self.j_mock = self.app.client_manager.data_processing.jobs
        self.j_mock.reset_mock()