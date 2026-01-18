from unittest import mock
from magnumclient.osc.v1 import quotas as osc_quotas
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestQuotasShow(TestQuotas):

    def setUp(self):
        super(TestQuotasShow, self).setUp()
        attr = dict()
        attr['name'] = 'fake-quota'
        attr['project_id'] = 'abc'
        attr['resource'] = 'Cluster'
        self._quota = magnum_fakes.FakeQuota.create_one_quota(attr)
        self._default_args = {'project_id': 'abc', 'resource': 'Cluster'}
        self.quotas_mock.get = mock.Mock()
        self.quotas_mock.get.return_value = self._quota
        self.cmd = osc_quotas.ShowQuotas(self.app, None)
        self.data = tuple(map(lambda x: getattr(self._quota, x), osc_quotas.QUOTA_ATTRIBUTES))

    def test_quotas_show(self):
        arglist = ['--project-id', 'abc', '--resource', 'Cluster']
        verifylist = [('project_id', 'abc'), ('resource', 'Cluster')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.quotas_mock.get.assert_called_with('abc', 'Cluster')

    def test_quotas_show_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)