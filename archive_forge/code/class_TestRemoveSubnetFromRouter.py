from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestRemoveSubnetFromRouter(TestRouter):
    """Remove subnet from Router"""
    _subnet = network_fakes.FakeSubnet.create_one_subnet()
    _router = network_fakes.FakeRouter.create_one_router(attrs={'subnet': _subnet.id})

    def setUp(self):
        super(TestRemoveSubnetFromRouter, self).setUp()
        self.network_client.find_router = mock.Mock(return_value=self._router)
        self.network_client.find_subnet = mock.Mock(return_value=self._subnet)
        self.cmd = router.RemoveSubnetFromRouter(self.app, self.namespace)

    def test_remove_subnet_no_option(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_remove_subnet_required_options(self):
        arglist = [self._router.id, self._router.subnet]
        verifylist = [('subnet', self._router.subnet), ('router', self._router.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.remove_interface_from_router.assert_called_with(self._router, **{'subnet_id': self._router.subnet})
        self.assertIsNone(result)