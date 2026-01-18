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
def _get_service_data(self, type=None, name=None, description=None, enabled=True):
    service_id = uuid.uuid4().hex
    name = name or uuid.uuid4().hex
    type = type or uuid.uuid4().hex
    response = {'id': service_id, 'name': name, 'type': type, 'enabled': enabled}
    if description is not None:
        response['description'] = description
    request = response.copy()
    request.pop('id')
    return _ServiceData(service_id, name, type, description, enabled, {'service': response}, {'OS-KSADM:service': response}, request)