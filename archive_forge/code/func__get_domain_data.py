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
def _get_domain_data(self, domain_name=None, description=None, enabled=None):
    domain_id = uuid.uuid4().hex
    domain_name = domain_name or self.getUniqueString('domainName')
    response = {'id': domain_id, 'name': domain_name}
    request = {'name': domain_name}
    if enabled is not None:
        request['enabled'] = bool(enabled)
        response['enabled'] = bool(enabled)
    if description:
        response['description'] = description
        request['description'] = description
    response.setdefault('enabled', True)
    return _DomainData(domain_id, domain_name, description, {'domain': response}, {'domain': request})