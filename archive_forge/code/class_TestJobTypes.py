from unittest import mock
from saharaclient.api import job_types as api_jt
from saharaclient.api.v2 import job_templates as api_job_templates
from saharaclient.osc.v2 import job_types as osc_jt
from saharaclient.tests.unit.osc.v1 import test_job_types as tjt_v1
class TestJobTypes(tjt_v1.TestJobTypes):

    def setUp(self):
        super(TestJobTypes, self).setUp()
        self.app.api_version['data_processing'] = '2'
        self.job_template_mock = self.app.client_manager.data_processing.job_templates
        self.jt_mock = self.app.client_manager.data_processing.job_types
        self.jt_mock.reset_mock()
        self.job_template_mock.reset_mock()