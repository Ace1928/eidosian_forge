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
class TestHelpHandling(base.TestBase):

    def _test_help(self, deferred_help):
        app, _ = make_app(deferred_help=deferred_help)
        with mock.patch.object(app, 'initialize_app') as init:
            with mock.patch('cliff.help.HelpAction.__call__', side_effect=SystemExit(0)) as helper:
                self.assertRaises(SystemExit, app.run, ['--help'])
                self.assertTrue(helper.called)
            self.assertEqual(deferred_help, init.called)

    def test_help(self):
        self._test_help(False)

    def test_deferred_help(self):
        self._test_help(True)

    def _test_interrupted_help(self, deferred_help):
        app, _ = make_app(deferred_help=deferred_help)
        with mock.patch('cliff.help.HelpAction.__call__', side_effect=KeyboardInterrupt):
            result = app.run(['--help'])
            self.assertEqual(result, 130)

    def test_interrupted_help(self):
        self._test_interrupted_help(False)

    def test_interrupted_deferred_help(self):
        self._test_interrupted_help(True)

    def _test_pipeclose_help(self, deferred_help):
        app, _ = make_app(deferred_help=deferred_help)
        with mock.patch('cliff.help.HelpAction.__call__', side_effect=BrokenPipeError):
            app.run(['--help'])

    def test_pipeclose_help(self):
        self._test_pipeclose_help(False)

    def test_pipeclose_deferred_help(self):
        self._test_pipeclose_help(True)

    def test_subcommand_help(self):
        app, _ = make_app(deferred_help=False)
        with mock.patch('cliff.help.HelpAction.__call__') as helper:
            app.run(['show', 'files', '--help'])
        self.assertTrue(helper.called)

    def test_subcommand_deferred_help(self):
        app, _ = make_app(deferred_help=True)
        with mock.patch.object(app, 'run_subcommand') as helper:
            app.run(['show', 'files', '--help'])
        helper.assert_called_once_with(['help', 'show', 'files'])