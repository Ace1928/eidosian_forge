import copy
import os
import sys
from unittest import mock
import testtools
from osc_lib import shell
from osc_lib.tests import utils
class TestShellOptions(utils.TestShell):
    """Test the option handling by argparse and openstack.config.loader

    This covers getting the CLI options through the initial processing
    and validates the arguments to initialize_app() and occ_get_one()
    """

    def setUp(self):
        super(TestShellOptions, self).setUp()
        self.useFixture(utils.EnvFixture())

    def test_empty_auth(self):
        os.environ = {}
        self._assert_initialize_app_arg('', {})
        self._assert_cloud_region_arg('', {})

    def test_no_options(self):
        os.environ = {}
        self._assert_initialize_app_arg('', {})
        self._assert_cloud_region_arg('', {})

    def test_global_options(self):
        self._test_options_init_app(global_options)
        self._test_options_get_one_cloud(global_options)

    def test_global_env(self):
        self._test_env_init_app(global_options)
        self._test_env_get_one_cloud(global_options)