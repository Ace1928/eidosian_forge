import collections
import os
import tempfile
import time
import urllib
import uuid
import fixtures
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from requests import structures
from requests_mock.contrib import fixture as rm_fixture
import openstack.cloud
import openstack.config as occ
import openstack.connection
from openstack.fixture import connection as os_fixture
from openstack.tests import base
from openstack.tests import fakes
def _make_test_cloud(self, cloud_name='_test_cloud_', **kwargs):
    test_cloud = os.environ.get('OPENSTACKSDK_OS_CLOUD', cloud_name)
    self.cloud_config = self.config.get_one(cloud=test_cloud, validate=True, **kwargs)
    self.cloud = openstack.connection.Connection(config=self.cloud_config, strict=self.strict_cloud)