from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api.v2 import job_templates as api_j
from saharaclient.osc.v2 import job_templates as osc_j
from saharaclient.tests.unit.osc.v1 import test_job_templates as tjt_v1
class TestJobTemplates(tjt_v1.TestJobTemplates):

    def setUp(self):
        super(TestJobTemplates, self).setUp()
        self.app.api_version['data_processing'] = '2'
        self.job_mock = self.app.client_manager.data_processing.job_templates
        self.job_mock.reset_mock()