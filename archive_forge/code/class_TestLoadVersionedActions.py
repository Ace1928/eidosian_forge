import argparse
import io
import json
import re
import sys
from unittest import mock
import ddt
import fixtures
import keystoneauth1.exceptions as ks_exc
from keystoneauth1.exceptions import DiscoveryFailure
from keystoneauth1.identity.generic.password import Password as ks_password
from keystoneauth1 import session
import requests_mock
from testtools import matchers
import cinderclient
from cinderclient import api_versions
from cinderclient.contrib import noauth
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit import fake_actions_module
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
class TestLoadVersionedActions(utils.TestCase):

    def setUp(self):
        super(TestLoadVersionedActions, self).setUp()
        self.mock_completion()

    def test_load_versioned_actions_v3_0(self):
        parser = cinderclient.shell.CinderClientArgumentParser()
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        shell = cinderclient.shell.OpenStackCinderShell()
        shell.subcommands = {}
        shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('3.0'), False, [])
        self.assertIn('fake-action', shell.subcommands.keys())
        self.assertEqual('fake_action 3.0 to 3.1', shell.subcommands['fake-action'].get_default('func')())

    def test_load_versioned_actions_v3_2(self):
        parser = cinderclient.shell.CinderClientArgumentParser()
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        shell = cinderclient.shell.OpenStackCinderShell()
        shell.subcommands = {}
        shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('3.2'), False, [])
        self.assertIn('fake-action', shell.subcommands.keys())
        self.assertEqual('fake_action 3.2 to 3.3', shell.subcommands['fake-action'].get_default('func')())
        self.assertIn('fake-action2', shell.subcommands.keys())
        self.assertEqual('fake_action2', shell.subcommands['fake-action2'].get_default('func')())

    def test_load_versioned_actions_not_in_version_range(self):
        parser = cinderclient.shell.CinderClientArgumentParser()
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        shell = cinderclient.shell.OpenStackCinderShell()
        shell.subcommands = {}
        shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('3.10000'), False, [])
        self.assertNotIn('fake-action', shell.subcommands.keys())
        self.assertIn('fake-action2', shell.subcommands.keys())

    def test_load_versioned_actions_unsupported_input(self):
        parser = cinderclient.shell.CinderClientArgumentParser()
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        shell = cinderclient.shell.OpenStackCinderShell()
        shell.subcommands = {}
        self.assertRaises(exceptions.UnsupportedAttribute, shell._find_actions, subparsers, fake_actions_module, api_versions.APIVersion('3.6'), False, ['another-fake-action', '--foo'])

    def test_load_versioned_actions_with_help(self):
        parser = cinderclient.shell.CinderClientArgumentParser()
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        shell = cinderclient.shell.OpenStackCinderShell()
        shell.subcommands = {}
        with mock.patch.object(subparsers, 'add_parser') as mock_add_parser:
            shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('3.1'), True, [])
            self.assertIn('fake-action', shell.subcommands.keys())
            expected_help = 'help message (Supported by API versions %(start)s - %(end)s)' % {'start': '3.0', 'end': '3.3'}
            expected_desc = 'help message\n\n    This will not show up in help message\n    '
            mock_add_parser.assert_any_call('fake-action', help=expected_help, description=expected_desc, add_help=False, formatter_class=cinderclient.shell.OpenStackHelpFormatter)

    def test_load_versioned_actions_with_help_on_latest(self):
        parser = cinderclient.shell.CinderClientArgumentParser()
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        shell = cinderclient.shell.OpenStackCinderShell()
        shell.subcommands = {}
        with mock.patch.object(subparsers, 'add_parser') as mock_add_parser:
            shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('3.latest'), True, [])
            self.assertIn('another-fake-action', shell.subcommands.keys())
            expected_help = ' (Supported by API versions %(start)s - %(end)s)%(hint)s' % {'start': '3.6', 'end': '3.latest', 'hint': cinderclient.shell.HINT_HELP_MSG}
            mock_add_parser.assert_any_call('another-fake-action', help=expected_help, description='', add_help=False, formatter_class=cinderclient.shell.OpenStackHelpFormatter)

    @mock.patch.object(cinderclient.shell.CinderClientArgumentParser, 'add_argument')
    def test_load_versioned_actions_with_args(self, mock_add_arg):
        parser = cinderclient.shell.CinderClientArgumentParser(add_help=False)
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        shell = cinderclient.shell.OpenStackCinderShell()
        shell.subcommands = {}
        shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('3.1'), False, [])
        self.assertIn('fake-action2', shell.subcommands.keys())
        mock_add_arg.assert_has_calls([mock.call('-h', '--help', action='help', help='==SUPPRESS=='), mock.call('--foo')])

    @mock.patch.object(cinderclient.shell.CinderClientArgumentParser, 'add_argument')
    def test_load_versioned_actions_with_args2(self, mock_add_arg):
        parser = cinderclient.shell.CinderClientArgumentParser(add_help=False)
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        shell = cinderclient.shell.OpenStackCinderShell()
        shell.subcommands = {}
        shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('3.4'), False, [])
        self.assertIn('fake-action2', shell.subcommands.keys())
        mock_add_arg.assert_has_calls([mock.call('-h', '--help', action='help', help='==SUPPRESS=='), mock.call('--bar', help='bar help')])

    @mock.patch.object(cinderclient.shell.CinderClientArgumentParser, 'add_argument')
    def test_load_versioned_actions_with_args_not_in_version_range(self, mock_add_arg):
        parser = cinderclient.shell.CinderClientArgumentParser(add_help=False)
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        shell = cinderclient.shell.OpenStackCinderShell()
        shell.subcommands = {}
        shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('3.10000'), False, [])
        self.assertIn('fake-action2', shell.subcommands.keys())
        mock_add_arg.assert_has_calls([mock.call('-h', '--help', action='help', help='==SUPPRESS==')])

    @mock.patch.object(cinderclient.shell.CinderClientArgumentParser, 'add_argument')
    def test_load_versioned_actions_with_args_and_help(self, mock_add_arg):
        parser = cinderclient.shell.CinderClientArgumentParser(add_help=False)
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        shell = cinderclient.shell.OpenStackCinderShell()
        shell.subcommands = {}
        shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('3.4'), True, [])
        mock_add_arg.assert_has_calls([mock.call('-h', '--help', action='help', help='==SUPPRESS=='), mock.call('--bar', help='bar help (Supported by API versions 3.3 - 3.4)')])

    @mock.patch.object(cinderclient.shell.CinderClientArgumentParser, 'add_argument')
    def test_load_actions_with_versioned_args_v36(self, mock_add_arg):
        parser = cinderclient.shell.CinderClientArgumentParser(add_help=False)
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        shell = cinderclient.shell.OpenStackCinderShell()
        shell.subcommands = {}
        shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('3.6'), False, [])
        self.assertIn(mock.call('--foo', help='first foo'), mock_add_arg.call_args_list)
        self.assertNotIn(mock.call('--foo', help='second foo'), mock_add_arg.call_args_list)

    @mock.patch.object(cinderclient.shell.CinderClientArgumentParser, 'add_argument')
    def test_load_actions_with_versioned_args_v39(self, mock_add_arg):
        parser = cinderclient.shell.CinderClientArgumentParser(add_help=False)
        subparsers = parser.add_subparsers(metavar='<subcommand>')
        shell = cinderclient.shell.OpenStackCinderShell()
        shell.subcommands = {}
        shell._find_actions(subparsers, fake_actions_module, api_versions.APIVersion('3.9'), False, [])
        self.assertNotIn(mock.call('--foo', help='first foo'), mock_add_arg.call_args_list)
        self.assertIn(mock.call('--foo', help='second foo'), mock_add_arg.call_args_list)