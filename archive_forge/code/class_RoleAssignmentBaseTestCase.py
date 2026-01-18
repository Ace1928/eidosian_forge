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
class RoleAssignmentBaseTestCase(test_v3.RestfulTestCase, test_v3.AssignmentTestMixin):
    """Base class for testing /v3/role_assignments API behavior."""
    MAX_HIERARCHY_BREADTH = 3
    MAX_HIERARCHY_DEPTH = CONF.max_project_tree_depth - 1

    def load_sample_data(self):
        """Create sample data to be used on tests.

        Created data are i) a role and ii) a domain containing: a project
        hierarchy and 3 users within 3 groups.

        """

        def create_project_hierarchy(parent_id, depth):
            """Create a random project hierarchy."""
            if depth == 0:
                return
            breadth = random.randint(1, self.MAX_HIERARCHY_BREADTH)
            subprojects = []
            for i in range(breadth):
                subprojects.append(unit.new_project_ref(domain_id=self.domain_id, parent_id=parent_id))
                PROVIDERS.resource_api.create_project(subprojects[-1]['id'], subprojects[-1])
            new_parent = subprojects[random.randint(0, breadth - 1)]
            create_project_hierarchy(new_parent['id'], depth - 1)
        super(RoleAssignmentBaseTestCase, self).load_sample_data()
        self.domain = unit.new_domain_ref()
        self.domain_id = self.domain['id']
        PROVIDERS.resource_api.create_domain(self.domain_id, self.domain)
        self.project = unit.new_project_ref(domain_id=self.domain_id)
        self.project_id = self.project['id']
        PROVIDERS.resource_api.create_project(self.project_id, self.project)
        create_project_hierarchy(self.project_id, random.randint(1, self.MAX_HIERARCHY_DEPTH))
        self.user_ids = []
        for i in range(3):
            user = unit.new_user_ref(domain_id=self.domain_id)
            user = PROVIDERS.identity_api.create_user(user)
            self.user_ids.append(user['id'])
        self.group_ids = []
        for i in range(3):
            group = unit.new_group_ref(domain_id=self.domain_id)
            group = PROVIDERS.identity_api.create_group(group)
            self.group_ids.append(group['id'])
            PROVIDERS.identity_api.add_user_to_group(user_id=self.user_ids[i], group_id=group['id'])
            PROVIDERS.identity_api.add_user_to_group(user_id=self.user_ids[i % 2], group_id=group['id'])
        PROVIDERS.assignment_api.create_grant(user_id=self.user_id, project_id=self.project_id, role_id=self.role_id)
        self.role = unit.new_role_ref()
        self.role_id = self.role['id']
        PROVIDERS.role_api.create_role(self.role_id, self.role)
        self.default_user_id = self.user_ids[0]
        self.default_group_id = self.group_ids[0]

    def get_role_assignments(self, expected_status=http.client.OK, **filters):
        """Return the result from querying role assignment API + queried URL.

        Calls GET /v3/role_assignments?<params> and returns its result, where
        <params> is the HTTP query parameters form of effective option plus
        filters, if provided. Queried URL is returned as well.

        :returns: a tuple containing the list role assignments API response and
                  queried URL.

        """
        query_url = self._get_role_assignments_query_url(**filters)
        response = self.get(query_url, expected_status=expected_status)
        return (response, query_url)

    def _get_role_assignments_query_url(self, **filters):
        """Return non-effective role assignments query URL from given filters.

        :param filters: query parameters are created with the provided filters
                        on role assignments attributes. Valid filters are:
                        role_id, domain_id, project_id, group_id, user_id and
                        inherited_to_projects.

        :returns: role assignments query URL.

        """
        return self.build_role_assignment_query_url(**filters)