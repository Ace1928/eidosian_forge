from unittest import mock
from oslotest import base
from requests_mock.contrib import fixture
class BaseCommandTest(base.BaseTestCase):

    def setUp(self):
        super(BaseCommandTest, self).setUp()
        self.app = mock.Mock()
        self.client = self.app.client_manager.workflow_engine

    def call(self, command, app_args=(), prog_name=''):
        cmd = command(self.app, app_args)
        parsed_args = cmd.get_parser(prog_name).parse_args(app_args)
        return cmd.take_action(parsed_args)