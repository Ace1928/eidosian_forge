from osc_lib.tests import utils as osc_u
import testtools
from unittest import mock
from saharaclient.api import job_binaries as api_jb
from saharaclient.osc.v1 import job_binaries as osc_jb
from saharaclient.tests.unit.osc.v1 import test_job_binaries as tjb_v1
class TestShowJobBinary(TestJobBinaries):

    def setUp(self):
        super(TestShowJobBinary, self).setUp()
        self.jb_mock.find_unique.return_value = api_jb.JobBinaries(None, JOB_BINARY_INFO)
        self.cmd = osc_jb.ShowJobBinary(self.app, None)

    def test_job_binary_show(self):
        arglist = ['job-binary']
        verifylist = [('job_binary', 'job-binary')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.jb_mock.find_unique.assert_called_once_with(name='job-binary')
        expected_columns = ('Description', 'Id', 'Is protected', 'Is public', 'Name', 'Url')
        self.assertEqual(expected_columns, columns)
        expected_data = ('descr', 'jb_id', False, False, 'job-binary', 'swift://cont/test')
        self.assertEqual(expected_data, data)