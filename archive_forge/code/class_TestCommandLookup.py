import argparse
import codecs
import io
from unittest import mock
from cliff import app as application
from cliff import command as c_cmd
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils as test_utils
from cliff import utils
import sys
class TestCommandLookup(base.TestBase):

    def test_unknown_cmd(self):
        app, command = make_app()
        self.assertEqual(2, app.run(['hell']))

    def test_unknown_cmd_debug(self):
        app, command = make_app()
        try:
            self.assertEqual(2, app.run(['--debug', 'hell']))
        except ValueError as err:
            self.assertIn("['hell']", str(err))

    def test_list_matching_commands(self):
        stdout = io.StringIO()
        app = application.App('testing', '1', test_utils.TestCommandManager(test_utils.TEST_NAMESPACE), stdout=stdout)
        app.NAME = 'test'
        try:
            self.assertEqual(2, app.run(['t']))
        except SystemExit:
            pass
        output = stdout.getvalue()
        self.assertIn("test: 't' is not a test command. See 'test --help'.", output)
        self.assertIn('Did you mean one of these?', output)
        self.assertIn('three word command\n  two words\n', output)

    def test_fuzzy_no_commands(self):
        cmd_mgr = commandmanager.CommandManager('cliff.fuzzy')
        app = application.App('test', '1.0', cmd_mgr)
        cmd_mgr.commands = {}
        matches = app.get_fuzzy_matches('foo')
        self.assertEqual([], matches)

    def test_fuzzy_common_prefix(self):
        cmd_mgr = commandmanager.CommandManager('cliff.fuzzy')
        app = application.App('test', '1.0', cmd_mgr)
        cmd_mgr.commands = {}
        cmd_mgr.add_command('user list', test_utils.TestCommand)
        cmd_mgr.add_command('user show', test_utils.TestCommand)
        matches = app.get_fuzzy_matches('user')
        self.assertEqual(['user list', 'user show'], matches)

    def test_fuzzy_same_distance(self):
        cmd_mgr = commandmanager.CommandManager('cliff.fuzzy')
        app = application.App('test', '1.0', cmd_mgr)
        cmd_mgr.add_command('user', test_utils.TestCommand)
        for cmd in cmd_mgr.commands.keys():
            self.assertEqual(8, utils.damerau_levenshtein('node', cmd, utils.COST))
        matches = app.get_fuzzy_matches('node')
        self.assertEqual(['complete', 'help', 'user'], matches)

    def test_fuzzy_no_prefix(self):
        cmd_mgr = commandmanager.CommandManager('cliff.fuzzy')
        app = application.App('test', '1.0', cmd_mgr)
        cmd_mgr.add_command('user', test_utils.TestCommand)
        matches = app.get_fuzzy_matches('uesr')
        self.assertEqual(['user'], matches)