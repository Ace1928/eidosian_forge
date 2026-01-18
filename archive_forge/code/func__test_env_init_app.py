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
def _test_env_init_app(self, test_opts):
    """Test options in the environment"""
    for opt in test_opts.keys():
        if not test_opts[opt][2]:
            continue
        key = opt2attr(opt)
        kwargs = {key: test_opts[opt][0]}
        env = {opt2env(opt): test_opts[opt][0]}
        os.environ = env.copy()
        self._assert_initialize_app_arg('', kwargs)