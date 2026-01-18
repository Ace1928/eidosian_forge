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
def _assert_cloud_region_arg(self, cmd_options, default_args):
    """Check the args passed to OpenStackConfig.get_one()

        The argparse argument to get_one() is an argparse.Namespace
        object that contains all of the options processed to this point in
        initialize_app().
        """
    cloud = mock.Mock(name='cloudy')
    cloud.config = {}
    self.occ_get_one = mock.Mock(return_value=cloud)
    with mock.patch('openstack.config.loader.OpenStackConfig.get_one', self.occ_get_one):
        _shell = make_shell(shell_class=self.shell_class)
        _cmd = cmd_options + ' module list'
        fake_execute(_shell, _cmd)
        self.app.assert_called_with(['module', 'list'])
        opts = self.occ_get_one.call_args[1]['argparse']
        for k in default_args.keys():
            self.assertEqual(default_args[k], vars(opts)[k], '%s does not match' % k)