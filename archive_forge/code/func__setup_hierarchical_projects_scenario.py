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
def _setup_hierarchical_projects_scenario(self):
    """Create basic hierarchical projects scenario.

        This basic scenario contains a root with one leaf project and
        two roles with the following names: non-inherited and inherited.

        """
    root = unit.new_project_ref(domain_id=self.domain['id'])
    leaf = unit.new_project_ref(domain_id=self.domain['id'], parent_id=root['id'])
    PROVIDERS.resource_api.create_project(root['id'], root)
    PROVIDERS.resource_api.create_project(leaf['id'], leaf)
    non_inherited_role = unit.new_role_ref(name='non-inherited')
    PROVIDERS.role_api.create_role(non_inherited_role['id'], non_inherited_role)
    inherited_role = unit.new_role_ref(name='inherited')
    PROVIDERS.role_api.create_role(inherited_role['id'], inherited_role)
    return (root['id'], leaf['id'], non_inherited_role['id'], inherited_role['id'])