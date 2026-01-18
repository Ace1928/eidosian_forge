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
def _test_pipeclose_help(self, deferred_help):
    app, _ = make_app(deferred_help=deferred_help)
    with mock.patch('cliff.help.HelpAction.__call__', side_effect=BrokenPipeError):
        app.run(['--help'])