import os
import sys
import tempfile
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import fixture as keystone_fixture
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from requests_mock.contrib import fixture as rm_fixture
import testscenarios
import testtools
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
import heatclient.shell
from heatclient.tests.unit import fakes
import heatclient.v1.shell
class ShellTestCommon(ShellBase):

    def setUp(self):
        super(ShellTestCommon, self).setUp()
        self.client = http.SessionClient
        self.set_fake_env(FAKE_ENV_KEYSTONE_V2)

    def test_help_unknown_command(self):
        self.assertRaises(exc.CommandError, self.shell, 'help foofoo')

    def test_help(self):
        required = ['^usage: heat', '(?m)^See "heat help COMMAND" for help on a specific command']
        for argstr in ['--help', 'help']:
            help_text = self.shell(argstr)
            for r in required:
                self.assertRegex(help_text, r)

    def test_command_help(self):
        output = self.shell('help help')
        self.assertIn('usage: heat help [<subcommand>]', output)
        subcommands = list(self.subcommands)
        for command in subcommands:
            if command.replace('_', '-') == 'bash-completion':
                continue
            output1 = self.shell('help %s' % command)
            output2 = self.shell('%s --help' % command)
            self.assertEqual(output1, output2)
            self.assertRegex(output1, '^usage: heat %s' % command)

    def test_debug_switch_raises_error(self):
        self.register_keystone_auth_fixture()
        self.mock_request_error('/stacks?', 'GET', exc.Unauthorized('FAIL'))
        args = ['--debug', 'stack-list']
        self.assertRaises(exc.Unauthorized, heatclient.shell.main, args)

    def test_dash_d_switch_raises_error(self):
        self.register_keystone_auth_fixture()
        self.mock_request_error('/stacks?', 'GET', exc.CommandError('FAIL'))
        args = ['-d', 'stack-list']
        self.assertRaises(exc.CommandError, heatclient.shell.main, args)

    def test_no_debug_switch_no_raises_errors(self):
        self.register_keystone_auth_fixture()
        self.mock_request_error('/stacks?', 'GET', exc.Unauthorized('FAIL'))
        args = ['stack-list']
        self.assertRaises(SystemExit, heatclient.shell.main, args)

    def test_help_on_subcommand(self):
        required = ['^usage: heat stack-list', "(?m)^List the user's stacks"]
        argstrings = ['help stack-list']
        for argstr in argstrings:
            help_text = self.shell(argstr)
            for r in required:
                self.assertRegex(help_text, r)