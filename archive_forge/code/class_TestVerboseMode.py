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
class TestVerboseMode(base.TestBase):

    def test_verbose(self):
        app, command = make_app()
        app.clean_up = mock.MagicMock(name='clean_up')
        app.run(['--verbose', 'mock'])
        app.clean_up.assert_called_once_with(command.return_value, 0, None)
        app.clean_up.reset_mock()
        app.run(['--quiet', 'mock'])
        app.clean_up.assert_called_once_with(command.return_value, 0, None)
        self.assertRaises(SystemExit, app.run, ['--verbose', '--quiet', 'mock'])