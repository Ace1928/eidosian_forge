from unittest import mock
from magnumclient.osc.v1 import quotas as osc_quotas
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestQuotasCreate(TestQuotas):

    def setUp(self):
        super(TestQuotasCreate, self).setUp()
        attr = dict()
        attr['name'] = 'fake-quota'
        attr['project_id'] = 'abc'
        attr['resource'] = 'Cluster'
        self._quota = magnum_fakes.FakeQuota.create_one_quota(attr)
        self._default_args = {'project_id': 'abc', 'resource': 'Cluster', 'hard_limit': 1}
        self.quotas_mock.create = mock.Mock()
        self.quotas_mock.create.return_value = self._quota
        self.cmd = osc_quotas.CreateQuotas(self.app, None)
        self.data = tuple(map(lambda x: getattr(self._quota, x), osc_quotas.QUOTA_ATTRIBUTES))

    def test_quotas_create(self):
        arglist = ['--project-id', 'abc', '--resource', 'Cluster']
        verifylist = [('project_id', 'abc'), ('resource', 'Cluster')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.quotas_mock.create.assert_called_with(**self._default_args)

    def test_quotas_create_with_hardlimit(self):
        arglist = ['--project-id', 'abc', '--resource', 'Cluster', '--hard-limit', '10']
        verifylist = [('project_id', 'abc'), ('resource', 'Cluster'), ('hard_limit', 10)]
        self._default_args['hard_limit'] = 10
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.quotas_mock.create.assert_called_with(**self._default_args)

    def test_quotas_create_wrong_projectid(self):
        arglist = ['abcd']
        verifylist = [('project_id', 'abcd')]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)

    def test_quotas_create_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)

    def test_quotas_create_with_wrong_args(self):
        arglist = ['--project-id', 'abc', '--resources', 'Cluster', '--hard-limit', '10']
        verifylist = []
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)