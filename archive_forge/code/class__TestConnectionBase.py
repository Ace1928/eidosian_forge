import os
from unittest import mock
import fixtures
from keystoneauth1 import session
from testtools import matchers
import openstack.config
from openstack import connection
from openstack import proxy
from openstack import service_description
from openstack.tests import fakes
from openstack.tests.unit import base
from openstack.tests.unit.fake import fake_service
class _TestConnectionBase(base.TestCase):

    def setUp(self):
        super(_TestConnectionBase, self).setUp()
        config_dir = self.useFixture(fixtures.TempDir()).path
        config_path = os.path.join(config_dir, 'clouds.yaml')
        with open(config_path, 'w') as conf:
            conf.write(CLOUD_CONFIG)
        self.useFixture(fixtures.EnvironmentVariable('OS_CLIENT_CONFIG_FILE', config_path))
        self.use_keystone_v2()