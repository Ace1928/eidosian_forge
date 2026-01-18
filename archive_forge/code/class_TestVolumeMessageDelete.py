from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_message
class TestVolumeMessageDelete(TestVolumeMessage):
    fake_messages = volume_fakes.create_volume_messages(count=2)

    def setUp(self):
        super().setUp()
        self.volume_messages_mock.get = volume_fakes.get_volume_messages(self.fake_messages)
        self.volume_messages_mock.delete.return_value = None
        self.cmd = volume_message.DeleteMessage(self.app, None)

    def test_message_delete(self):
        self.volume_client.api_version = api_versions.APIVersion('3.3')
        arglist = [self.fake_messages[0].id]
        verifylist = [('message_ids', [self.fake_messages[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volume_messages_mock.delete.assert_called_with(self.fake_messages[0].id)
        self.assertIsNone(result)

    def test_message_delete_multiple_messages(self):
        self.volume_client.api_version = api_versions.APIVersion('3.3')
        arglist = [self.fake_messages[0].id, self.fake_messages[1].id]
        verifylist = [('message_ids', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for m in self.fake_messages:
            calls.append(call(m.id))
        self.volume_messages_mock.delete.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_message_delete_multiple_messages_with_exception(self):
        self.volume_client.api_version = api_versions.APIVersion('3.3')
        arglist = [self.fake_messages[0].id, 'invalid_message']
        verifylist = [('message_ids', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.volume_messages_mock.delete.side_effect = [self.fake_messages[0], exceptions.CommandError]
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertEqual('Failed to delete 1 of 2 messages.', str(exc))
        self.volume_messages_mock.delete.assert_any_call(self.fake_messages[0].id)
        self.volume_messages_mock.delete.assert_any_call('invalid_message')
        self.assertEqual(2, self.volume_messages_mock.delete.call_count)

    def test_message_delete_pre_v33(self):
        self.volume_client.api_version = api_versions.APIVersion('3.2')
        arglist = [self.fake_messages[0].id]
        verifylist = [('message_ids', [self.fake_messages[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.3 or greater is required', str(exc))