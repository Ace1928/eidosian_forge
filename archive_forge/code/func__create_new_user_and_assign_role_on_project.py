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
def _create_new_user_and_assign_role_on_project(self):
    """Create a new user and assign user a role on a project."""
    new_user = unit.new_user_ref(domain_id=self.domain_id)
    user_ref = PROVIDERS.identity_api.create_user(new_user)
    collection_url = '/projects/%(project_id)s/users/%(user_id)s/roles' % {'project_id': self.project_id, 'user_id': user_ref['id']}
    member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
    self.put(member_url)
    self.head(member_url)
    self.get(member_url, expected_status=http.client.NO_CONTENT)
    return (member_url, user_ref)