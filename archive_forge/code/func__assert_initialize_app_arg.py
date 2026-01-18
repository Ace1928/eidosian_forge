import contextlib
import copy
import json as jsonutils
import os
from unittest import mock
from cliff import columns as cliff_columns
import fixtures
from keystoneauth1 import loading
from openstack.config import cloud_region
from openstack.config import defaults
from oslo_utils import importutils
from requests_mock.contrib import fixture
import testtools
from osc_lib import clientmanager
from osc_lib import shell
from osc_lib.tests import fakes
def _assert_initialize_app_arg(self, cmd_options, default_args):
    """Check the args passed to initialize_app()

        The argv argument to initialize_app() is the remainder from parsing
        global options declared in both cliff.app and
        osc_lib.OpenStackShell build_option_parser().  Any global
        options passed on the command line should not be in argv but in
        _shell.options.
        """
    with mock.patch(self.shell_class_name + '.initialize_app', self.app):
        _shell = make_shell(shell_class=self.shell_class)
        _cmd = cmd_options + ' module list'
        fake_execute(_shell, _cmd)
        self.app.assert_called_with(['module', 'list'])
        for k in default_args.keys():
            self.assertEqual(default_args[k], vars(_shell.options)[k], '%s does not match' % k)