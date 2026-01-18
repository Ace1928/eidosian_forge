import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
from oslo_db import exception as db_exception
from oslo_db import options
from oslo_log import log
import sqlalchemy
from sqlalchemy import exc
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import sql
from keystone.common.sql import core
import keystone.conf
from keystone.credential.providers import fernet as credential_provider
from keystone import exception
from keystone.identity.backends import sql_model as identity_sql
from keystone.resource.backends import base as resource
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.limit import test_backends as limit_tests
from keystone.tests.unit.policy import test_backends as policy_tests
from keystone.tests.unit.resource import test_backends as resource_tests
from keystone.tests.unit.trust import test_backends as trust_tests
from keystone.trust.backends import sql as trust_sql
class SqlLimit(SqlTests, limit_tests.LimitTests):

    def setUp(self):
        super(SqlLimit, self).setUp()
        fixtures_to_cleanup = []
        for service in default_fixtures.SERVICES:
            service_id = service['id']
            rv = PROVIDERS.catalog_api.create_service(service_id, service)
            attrname = service['extra']['name']
            setattr(self, attrname, rv)
            fixtures_to_cleanup.append(attrname)
        for region in default_fixtures.REGIONS:
            rv = PROVIDERS.catalog_api.create_region(region)
            attrname = region['id']
            setattr(self, attrname, rv)
            fixtures_to_cleanup.append(attrname)
        self.addCleanup(self.cleanup_instance(*fixtures_to_cleanup))
        registered_limit_1 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_one['id'], resource_name='volume', default_limit=10, id=uuid.uuid4().hex)
        registered_limit_2 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_two['id'], resource_name='snapshot', default_limit=10, id=uuid.uuid4().hex)
        registered_limit_3 = unit.new_registered_limit_ref(service_id=self.service_one['id'], region_id=self.region_two['id'], resource_name='backup', default_limit=10, id=uuid.uuid4().hex)
        PROVIDERS.unified_limit_api.create_registered_limits([registered_limit_1, registered_limit_2, registered_limit_3])