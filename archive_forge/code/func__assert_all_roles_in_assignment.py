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
def _assert_all_roles_in_assignment(self, response, user):
    self.assertValidRoleAssignmentListResponse(response, expected_length=len(self.role_list), resource_url=self._build_effective_role_assignments_url(user))