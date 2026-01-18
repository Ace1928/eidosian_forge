from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_message
class TestVolumeMessageList(TestVolumeMessage):
    fake_project = identity_fakes.FakeProject.create_one_project()
    fake_messages = volume_fakes.create_volume_messages(count=3)
    columns = ('ID', 'Event ID', 'Resource Type', 'Resource UUID', 'Message Level', 'User Message', 'Request ID', 'Created At', 'Guaranteed Until')
    data = []
    for fake_message in fake_messages:
        data.append((fake_message.id, fake_message.event_id, fake_message.resource_type, fake_message.resource_uuid, fake_message.message_level, fake_message.user_message, fake_message.request_id, fake_message.created_at, fake_message.guaranteed_until))

    def setUp(self):
        super().setUp()
        self.projects_mock.get.return_value = self.fake_project
        self.volume_messages_mock.list.return_value = self.fake_messages
        self.cmd = volume_message.ListMessages(self.app, None)

    def test_message_list(self):
        self.volume_client.api_version = api_versions.APIVersion('3.3')
        arglist = []
        verifylist = [('project', None), ('marker', None), ('limit', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'project_id': None}
        self.volume_messages_mock.list.assert_called_with(search_opts=search_opts, marker=None, limit=None)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_message_list_with_options(self):
        self.volume_client.api_version = api_versions.APIVersion('3.3')
        arglist = ['--project', self.fake_project.name, '--marker', self.fake_messages[0].id, '--limit', '3']
        verifylist = [('project', self.fake_project.name), ('marker', self.fake_messages[0].id), ('limit', 3)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'project_id': self.fake_project.id}
        self.volume_messages_mock.list.assert_called_with(search_opts=search_opts, marker=self.fake_messages[0].id, limit=3)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, list(data))

    def test_message_list_pre_v33(self):
        self.volume_client.api_version = api_versions.APIVersion('3.2')
        arglist = []
        verifylist = [('project', None), ('marker', None), ('limit', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.3 or greater is required', str(exc))