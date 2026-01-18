from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api.v2 import jobs as api_j
from saharaclient.osc.v2 import jobs as osc_j
from saharaclient.tests.unit.osc.v1 import test_jobs as tj_v1
class TestUpdateJob(TestJobs):

    def setUp(self):
        super(TestUpdateJob, self).setUp()
        self.j_mock.get.return_value = api_j.Job(None, JOB_INFO)
        self.j_mock.update.return_value = mock.Mock(job=JOB_INFO.copy())
        self.cmd = osc_j.UpdateJob(self.app, None)

    def test_job_update_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_job_update_nothing_updated(self):
        arglist = ['job_id']
        verifylist = [('job', 'job_id')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.j_mock.update.assert_called_once_with('job_id')

    def test_job_update_public_protected(self):
        arglist = ['job_id', '--public', '--protected']
        verifylist = [('job', 'job_id'), ('is_public', True), ('is_protected', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.j_mock.update.assert_called_once_with('job_id', is_protected=True, is_public=True)
        expected_columns = ('Cluster id', 'End time', 'Engine job id', 'Id', 'Input id', 'Is protected', 'Is public', 'Job template id', 'Output id', 'Start time', 'Status')
        self.assertEqual(expected_columns, columns)
        expected_data = ('cluster_id', 'end', 'engine_job_id', 'j_id', 'input_id', False, False, 'job_template_id', 'output_id', 'start', 'SUCCEEDED')
        self.assertEqual(expected_data, data)

    def test_job_update_private_unprotected(self):
        arglist = ['job_id', '--private', '--unprotected']
        verifylist = [('job', 'job_id'), ('is_public', False), ('is_protected', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.j_mock.update.assert_called_once_with('job_id', is_protected=False, is_public=False)