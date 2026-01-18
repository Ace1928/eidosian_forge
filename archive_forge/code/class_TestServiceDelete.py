from keystoneclient import exceptions as identity_exc
from osc_lib import exceptions
from openstackclient.identity.v2_0 import service
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestServiceDelete(TestService):

    def setUp(self):
        super(TestServiceDelete, self).setUp()
        self.services_mock.get.side_effect = identity_exc.NotFound(None)
        self.services_mock.find.return_value = self.fake_service
        self.services_mock.delete.return_value = None
        self.cmd = service.DeleteService(self.app, None)

    def test_service_delete_no_options(self):
        arglist = [self.fake_service.name]
        verifylist = [('services', [self.fake_service.name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.services_mock.delete.assert_called_with(self.fake_service.id)
        self.assertIsNone(result)