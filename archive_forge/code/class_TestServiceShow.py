from keystoneclient import exceptions as identity_exc
from osc_lib import exceptions
from openstackclient.identity.v2_0 import service
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestServiceShow(TestService):
    fake_service_s = identity_fakes.FakeService.create_one_service()

    def setUp(self):
        super(TestServiceShow, self).setUp()
        self.services_mock.get.side_effect = identity_exc.NotFound(None)
        self.services_mock.find.return_value = self.fake_service_s
        self.cmd = service.ShowService(self.app, None)

    def test_service_show(self):
        arglist = [self.fake_service_s.name]
        verifylist = [('service', self.fake_service_s.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.services_mock.find.assert_called_with(name=self.fake_service_s.name)
        collist = ('description', 'id', 'name', 'type')
        self.assertEqual(collist, columns)
        datalist = (self.fake_service_s.description, self.fake_service_s.id, self.fake_service_s.name, self.fake_service_s.type)
        self.assertEqual(datalist, data)

    def test_service_show_nounique(self):
        self.services_mock.find.side_effect = identity_exc.NoUniqueMatch(None)
        arglist = ['nounique_service']
        verifylist = [('service', 'nounique_service')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual("Multiple service matches found for 'nounique_service', use an ID to be more specific.", str(e))