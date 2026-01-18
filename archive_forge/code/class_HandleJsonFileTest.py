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
class HandleJsonFileTest(test_utils.BaseTestCase):

    def test_handle_json_or_file_arg(self):
        cleansteps = '[{"step": "upgrade", "interface": "deploy"}]'
        steps = utils.handle_json_or_file_arg(cleansteps)
        self.assertEqual(json.loads(cleansteps), steps)

    def test_handle_json_or_file_arg_bad_json(self):
        cleansteps = '{foo invalid: json{'
        self.assertRaisesRegex(exc.InvalidAttribute, 'is not a file and cannot be parsed as JSON', utils.handle_json_or_file_arg, cleansteps)

    def test_handle_json_or_file_arg_file(self):
        contents = '[{"step": "upgrade", "interface": "deploy"}]'
        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(contents)
            f.flush()
            steps = utils.handle_json_or_file_arg(f.name)
        self.assertEqual(json.loads(contents), steps)

    def test_handle_yaml_or_file_arg_file(self):
        contents = '---\n- step: upgrade\n  interface: deploy'
        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write(contents)
            f.flush()
            steps = utils.handle_json_or_file_arg(f.name)
        self.assertEqual([{'step': 'upgrade', 'interface': 'deploy'}], steps)

    @mock.patch.object(builtins, 'open', autospec=True)
    def test_handle_json_or_file_arg_file_fail(self, mock_open):
        mock_open.return_value.__enter__.side_effect = IOError
        with tempfile.NamedTemporaryFile(mode='w') as f:
            self.assertRaisesRegex(exc.InvalidAttribute, 'from file', utils.handle_json_or_file_arg, f.name)
            mock_open.assert_called_once_with(f.name, 'r')