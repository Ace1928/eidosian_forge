from unittest import mock
from openstackclient.identity.v3 import endpoint_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestEndpointGroupSet(TestEndpointGroup):
    endpoint_group = identity_fakes.FakeEndpointGroup.create_one_endpointgroup()

    def setUp(self):
        super(TestEndpointGroupSet, self).setUp()
        self.endpoint_groups_mock.get.return_value = self.endpoint_group
        self.endpoint_groups_mock.update.return_value = self.endpoint_group
        self.cmd = endpoint_group.SetEndpointGroup(self.app, None)

    def test_endpoint_group_set_no_options(self):
        arglist = [self.endpoint_group.id]
        verifylist = [('endpointgroup', self.endpoint_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'name': None, 'filters': None, 'description': ''}
        self.endpoint_groups_mock.update.assert_called_with(self.endpoint_group.id, **kwargs)
        self.assertIsNone(result)

    def test_endpoint_group_set_name(self):
        arglist = ['--name', 'qwerty', self.endpoint_group.id]
        verifylist = [('name', 'qwerty'), ('endpointgroup', self.endpoint_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'name': 'qwerty', 'filters': None, 'description': ''}
        self.endpoint_groups_mock.update.assert_called_with(self.endpoint_group.id, **kwargs)
        self.assertIsNone(result)

    def test_endpoint_group_set_filters(self):
        arglist = ['--filters', identity_fakes.endpoint_group_file_path, self.endpoint_group.id]
        verifylist = [('filters', identity_fakes.endpoint_group_file_path), ('endpointgroup', self.endpoint_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        mocker = mock.Mock()
        mocker.return_value = identity_fakes.endpoint_group_filters_2
        with mock.patch('openstackclient.identity.v3.endpoint_group.SetEndpointGroup._read_filters', mocker):
            result = self.cmd.take_action(parsed_args)
        kwargs = {'name': None, 'filters': identity_fakes.endpoint_group_filters_2, 'description': ''}
        self.endpoint_groups_mock.update.assert_called_with(self.endpoint_group.id, **kwargs)
        self.assertIsNone(result)

    def test_endpoint_group_set_description(self):
        arglist = ['--description', 'qwerty', self.endpoint_group.id]
        verifylist = [('description', 'qwerty'), ('endpointgroup', self.endpoint_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'name': None, 'filters': None, 'description': 'qwerty'}
        self.endpoint_groups_mock.update.assert_called_with(self.endpoint_group.id, **kwargs)
        self.assertIsNone(result)