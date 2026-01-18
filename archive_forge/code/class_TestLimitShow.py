import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestLimitShow(TestLimit):

    def setUp(self):
        super(TestLimitShow, self).setUp()
        self.limit_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.LIMIT), loaded=True)
        self.cmd = limit.ShowLimit(self.app, None)

    def test_limit_show(self):
        arglist = [identity_fakes.limit_id]
        verifylist = [('limit_id', identity_fakes.limit_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.limit_mock.get.assert_called_with(identity_fakes.limit_id)
        collist = ('description', 'id', 'project_id', 'region_id', 'resource_limit', 'resource_name', 'service_id')
        self.assertEqual(collist, columns)
        datalist = (None, identity_fakes.limit_id, identity_fakes.project_id, None, identity_fakes.limit_resource_limit, identity_fakes.limit_resource_name, identity_fakes.service_id)
        self.assertEqual(datalist, data)