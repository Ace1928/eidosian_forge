from unittest import mock
from saharaclient.api import job_types as api_jt
from saharaclient.api.v2 import job_templates as api_job_templates
from saharaclient.osc.v2 import job_types as osc_jt
from saharaclient.tests.unit.osc.v1 import test_job_types as tjt_v1
class TestGetJobTypeConfigs(TestJobTypes):

    def setUp(self):
        super(TestGetJobTypeConfigs, self).setUp()
        self.job_template_mock.get_configs.return_value = api_job_templates.JobTemplate(None, JOB_TYPE_INFO)
        self.cmd = osc_jt.GetJobTypeConfigs(self.app, None)

    @mock.patch('oslo_serialization.jsonutils.dump')
    def test_get_job_type_configs_default_file(self, p_dump):
        m_open = mock.mock_open()
        with mock.patch('builtins.open', m_open, create=True):
            arglist = ['Pig']
            verifylist = [('job_type', 'Pig')]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            self.cmd.take_action(parsed_args)
            self.job_template_mock.get_configs.assert_called_once_with('Pig')
            args_to_dump = p_dump.call_args[0]
            self.assertEqual(JOB_TYPE_INFO, args_to_dump[0])
            self.assertEqual('Pig', m_open.call_args[0][0])

    @mock.patch('oslo_serialization.jsonutils.dump')
    def test_get_job_type_configs_specified_file(self, p_dump):
        m_open = mock.mock_open()
        with mock.patch('builtins.open', m_open):
            arglist = ['Pig', '--file', 'testfile']
            verifylist = [('job_type', 'Pig'), ('file', 'testfile')]
            parsed_args = self.check_parser(self.cmd, arglist, verifylist)
            self.cmd.take_action(parsed_args)
            self.job_template_mock.get_configs.assert_called_once_with('Pig')
            args_to_dump = p_dump.call_args[0]
            self.assertEqual(JOB_TYPE_INFO, args_to_dump[0])
            self.assertEqual('testfile', m_open.call_args[0][0])