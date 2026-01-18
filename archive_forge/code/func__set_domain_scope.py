import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def _set_domain_scope(self, domain_id):
    if CONF.identity.domain_specific_drivers_enabled:
        return domain_id