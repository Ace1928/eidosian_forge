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
class RoleAssignmentInheritedTestCase(RoleAssignmentDirectTestCase):
    """Class for testing inherited assignments on /v3/role_assignments API.

    Inherited assignments on a domain or project have no effect on them
    directly, but on the projects under them instead.

    Tests on this class do not make assertions on the effect of inherited
    assignments, but in their representation and API filtering.

    """

    def _test_get_role_assignments(self, **filters):
        """Add inherited_to_project filter to expected entity in tests."""
        super(RoleAssignmentInheritedTestCase, self)._test_get_role_assignments(inherited_to_projects=True, **filters)