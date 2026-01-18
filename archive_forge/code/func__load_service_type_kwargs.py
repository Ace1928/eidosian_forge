import testtools
from unittest import mock
from keystoneauth1.exceptions import catalog
from magnumclient.v1 import client
def _load_service_type_kwargs(self):
    return {'interface': 'public', 'region_name': None, 'service_name': None, 'service_type': 'container-infra'}