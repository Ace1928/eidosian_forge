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
class TestInteractiveMode(base.TestBase):

    def test_no_args_triggers_interactive_mode(self):
        app, command = make_app()
        app.interact = mock.MagicMock(name='inspect')
        app.run([])
        app.interact.assert_called_once_with()

    def test_interactive_mode_cmdloop(self):
        app, command = make_app()
        app.interactive_app_factory = mock.MagicMock(name='interactive_app_factory')
        self.assertIsNone(app.interpreter)
        ret = app.run([])
        self.assertIsNotNone(app.interpreter)
        cmdloop = app.interactive_app_factory.return_value.cmdloop
        cmdloop.assert_called_once_with()
        self.assertNotEqual(ret, 0)

    def test_interactive_mode_cmdloop_error(self):
        app, command = make_app()
        cmdloop_mock = mock.MagicMock(name='cmdloop')
        cmdloop_mock.return_value = 1
        app.interactive_app_factory = mock.MagicMock(name='interactive_app_factory')
        self.assertIsNone(app.interpreter)
        ret = app.run([])
        self.assertIsNotNone(app.interpreter)
        cmdloop = app.interactive_app_factory.return_value.cmdloop
        cmdloop.assert_called_once_with()
        self.assertNotEqual(ret, 0)