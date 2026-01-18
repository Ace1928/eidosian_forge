import copy
from openstackclient.identity.v3 import federation_protocol
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestProtocolDelete(TestProtocol):

    def setUp(self):
        super(TestProtocolDelete, self).setUp()
        self.protocols_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.PROTOCOL_OUTPUT), loaded=True)
        self.protocols_mock.delete.return_value = None
        self.cmd = federation_protocol.DeleteProtocol(self.app, None)

    def test_delete_identity_provider(self):
        arglist = ['--identity-provider', identity_fakes.idp_id, identity_fakes.protocol_id]
        verifylist = [('federation_protocol', [identity_fakes.protocol_id]), ('identity_provider', identity_fakes.idp_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.protocols_mock.delete.assert_called_with(identity_fakes.idp_id, identity_fakes.protocol_id)
        self.assertIsNone(result)