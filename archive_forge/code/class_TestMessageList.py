from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import messages as osc_messages
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestMessageList(TestMessage):

    def setUp(self):
        super(TestMessageList, self).setUp()
        self.messages = manila_fakes.FakeMessage.create_messages(count=2)
        self.messages_mock.list.return_value = self.messages
        self.values = (oscutils.get_dict_properties(m._info, COLUMNS) for m in self.messages)
        self.cmd = osc_messages.ListMessage(self.app, None)

    def test_list_messages(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.messages_mock.list.assert_called_with(search_opts={'limit': None, 'request_id': None, 'resource_type': None, 'resource_id': None, 'action_id': None, 'detail_id': None, 'message_level': None, 'created_since': None, 'created_before': None})
        self.assertEqual(COLUMNS, columns)
        self.assertEqual(list(self.values), list(data))

    def test_list_messages_api_version_exception(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.50')
        arglist = ['--before', '2021-02-06T09:49:58-05:00', '--since', '2021-02-05T09:49:58-05:00']
        verifylist = [('before', '2021-02-06T09:49:58-05:00'), ('since', '2021-02-05T09:49:58-05:00')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)