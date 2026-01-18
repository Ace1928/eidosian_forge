from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api.v2 import job_templates as api_j
from saharaclient.osc.v2 import job_templates as osc_j
from saharaclient.tests.unit.osc.v1 import test_job_templates as tjt_v1
class TestUpdateJobTemplate(TestJobTemplates):

    def setUp(self):
        super(TestUpdateJobTemplate, self).setUp()
        self.job_mock.find_unique.return_value = api_j.JobTemplate(None, JOB_TEMPLATE_INFO)
        self.job_mock.update.return_value = mock.Mock(job_template=JOB_TEMPLATE_INFO.copy())
        self.cmd = osc_j.UpdateJobTemplate(self.app, None)

    def test_job_template_update_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_job_template_update_nothing_updated(self):
        arglist = ['pig-job']
        verifylist = [('job_template', 'pig-job')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.job_mock.update.assert_called_once_with('job_id')

    def test_job_template_update_all_options(self):
        arglist = ['pig-job', '--name', 'pig-job', '--description', 'descr', '--public', '--protected']
        verifylist = [('job_template', 'pig-job'), ('name', 'pig-job'), ('description', 'descr'), ('is_public', True), ('is_protected', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.job_mock.update.assert_called_once_with('job_id', description='descr', is_protected=True, is_public=True, name='pig-job')
        expected_columns = ('Description', 'Id', 'Is protected', 'Is public', 'Libs', 'Mains', 'Name', 'Type')
        self.assertEqual(expected_columns, columns)
        expected_data = ('Job for test', 'job_id', False, False, 'lib:lib_id', 'main:main_id', 'pig-job', 'Pig')
        self.assertEqual(expected_data, data)

    def test_job_template_update_private_unprotected(self):
        arglist = ['pig-job', '--private', '--unprotected']
        verifylist = [('job_template', 'pig-job'), ('is_public', False), ('is_protected', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.job_mock.update.assert_called_once_with('job_id', is_protected=False, is_public=False)