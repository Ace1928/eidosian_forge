import copy
from unittest import mock
import uuid
from testtools import matchers
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
def _create_endpoints(self):

    def create_endpoint(service_id, region, **kwargs):
        ref = unit.new_endpoint_ref(service_id=service_id, region_id=region, url='http://localhost/%s' % uuid.uuid4().hex, **kwargs)
        PROVIDERS.catalog_api.create_endpoint(ref['id'], ref)
        return ref
    service_ref = unit.new_service_ref()
    service_id = service_ref['id']
    PROVIDERS.catalog_api.create_service(service_id, service_ref)
    region = unit.new_region_ref()
    PROVIDERS.catalog_api.create_region(region)
    enabled_endpoint_ref = create_endpoint(service_id, region['id'])
    disabled_endpoint_ref = create_endpoint(service_id, region['id'], enabled=False, interface='internal')
    return (service_ref, enabled_endpoint_ref, disabled_endpoint_ref)