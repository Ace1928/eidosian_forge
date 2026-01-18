import importlib
import os
from unittest import mock
from osc_lib.tests import utils as osc_lib_test_utils
import wrapt
from openstackclient import shell
def _assert_cli(self, cmd_options, default_args):
    with mock.patch(self.shell_class_name + '.initialize_app', self.app):
        _shell = osc_lib_test_utils.make_shell(shell_class=self.shell_class)
        _cmd = cmd_options + ' list server'
        osc_lib_test_utils.fake_execute(_shell, _cmd)
        self.app.assert_called_with(['list', 'server'])
        self.assertEqual(default_args['compute_api_version'], _shell.options.os_compute_api_version)
        self.assertEqual(default_args['identity_api_version'], _shell.options.os_identity_api_version)
        self.assertEqual(default_args['image_api_version'], _shell.options.os_image_api_version)
        self.assertEqual(default_args['volume_api_version'], _shell.options.os_volume_api_version)
        self.assertEqual(default_args['network_api_version'], _shell.options.os_network_api_version)