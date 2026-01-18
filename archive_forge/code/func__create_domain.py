from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def _create_domain(domain_id=None):
    if domain_id is None:
        new_domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
        return new_domain
    else:
        return PROVIDERS.resource_api.get_domain(domain_id)