from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestAddPortToRouter(TestRouter):
    """Add port to Router"""
    _port = network_fakes.create_one_port()
    _router = network_fakes.FakeRouter.create_one_router(attrs={'port': _port.id})

    def setUp(self):
        super(TestAddPortToRouter, self).setUp()
        self.network_client.find_router = mock.Mock(return_value=self._router)
        self.network_client.find_port = mock.Mock(return_value=self._port)
        self.cmd = router.AddPortToRouter(self.app, self.namespace)

    def test_add_port_no_option(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_add_port_required_options(self):
        arglist = [self._router.id, self._router.port]
        verifylist = [('router', self._router.id), ('port', self._router.port)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.add_interface_to_router.assert_called_with(self._router, **{'port_id': self._router.port})
        self.assertIsNone(result)