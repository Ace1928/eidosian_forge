import copy
from openstackclient.identity.v3 import federation_protocol
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestProtocolCreate(TestProtocol):

    def setUp(self):
        super(TestProtocolCreate, self).setUp()
        proto = copy.deepcopy(identity_fakes.PROTOCOL_OUTPUT)
        resource = fakes.FakeResource(None, proto, loaded=True)
        self.protocols_mock.create.return_value = resource
        self.cmd = federation_protocol.CreateProtocol(self.app, None)

    def test_create_protocol(self):
        argslist = [identity_fakes.protocol_id, '--identity-provider', identity_fakes.idp_id, '--mapping', identity_fakes.mapping_id]
        verifylist = [('federation_protocol', identity_fakes.protocol_id), ('identity_provider', identity_fakes.idp_id), ('mapping', identity_fakes.mapping_id)]
        parsed_args = self.check_parser(self.cmd, argslist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.protocols_mock.create.assert_called_with(protocol_id=identity_fakes.protocol_id, identity_provider=identity_fakes.idp_id, mapping=identity_fakes.mapping_id)
        collist = ('id', 'identity_provider', 'mapping')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.protocol_id, identity_fakes.idp_id, identity_fakes.mapping_id)
        self.assertEqual(datalist, data)