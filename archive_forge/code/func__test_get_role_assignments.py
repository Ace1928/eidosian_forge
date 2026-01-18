import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def _test_get_role_assignments(self, **filters):
    """Add inherited_to_project filter to expected entity in tests."""
    super(RoleAssignmentInheritedTestCase, self)._test_get_role_assignments(inherited_to_projects=True, **filters)