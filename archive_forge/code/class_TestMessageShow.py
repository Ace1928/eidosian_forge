from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import messages as osc_messages
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestMessageShow(TestMessage):

    def setUp(self):
        super(TestMessageShow, self).setUp()
        self.message = manila_fakes.FakeMessage.create_one_message()
        self.messages_mock.get.return_value = self.message
        self.cmd = osc_messages.ShowMessage(self.app, None)
        self.data = self.message._info.values()
        self.columns = self.message._info.keys()

    def test_message_show_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_message_show(self):
        arglist = [self.message.id]
        verifylist = [('message', self.message.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.messages_mock.get.assert_called_with(self.message.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)