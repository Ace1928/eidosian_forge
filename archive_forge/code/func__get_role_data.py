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
def _get_role_data(self, role_name=None):
    role_id = uuid.uuid4().hex
    role_name = role_name or uuid.uuid4().hex
    request = {'name': role_name}
    response = request.copy()
    response['id'] = role_id
    return _RoleData(role_id, role_name, {'role': response}, {'role': request})