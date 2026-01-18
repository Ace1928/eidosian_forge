import builtins
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.tests.unit import utils as test_utils
class GetFromStdinTest(test_utils.BaseTestCase):

    @mock.patch.object(sys, 'stdin', autospec=True)
    def test_get_from_stdin(self, mock_stdin):
        contents = '[{"step": "upgrade", "interface": "deploy"}]'
        mock_stdin.read.return_value = contents
        desc = 'something'
        info = utils.get_from_stdin(desc)
        self.assertEqual(info, contents)
        mock_stdin.read.assert_called_once_with()

    @mock.patch.object(sys, 'stdin', autospec=True)
    def test_get_from_stdin_fail(self, mock_stdin):
        mock_stdin.read.side_effect = IOError
        desc = 'something'
        self.assertRaises(exc.InvalidAttribute, utils.get_from_stdin, desc)
        mock_stdin.read.assert_called_once_with()