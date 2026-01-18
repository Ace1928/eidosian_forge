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
class RoleAssignmentEffectiveTestCase(RoleAssignmentInheritedTestCase):
    """Class for testing inheritance effects on /v3/role_assignments API.

    Inherited assignments on a domain or project have no effect on them
    directly, but on the projects under them instead.

    Tests on this class make assertions on the effect of inherited assignments
    and API filtering.

    """

    def _get_role_assignments_query_url(self, **filters):
        """Return effective role assignments query URL from given filters.

        For test methods in this class, effetive will always be true. As in
        effective mode, inherited_to_projects, group_id, domain_id and
        project_id will always be desconsidered from provided filters.

        :param filters: query parameters are created with the provided filters.
                        Valid filters are: role_id, domain_id, project_id,
                        group_id, user_id and inherited_to_projects.

        :returns: role assignments query URL.

        """
        query_filters = filters.copy()
        query_filters.pop('inherited_to_projects')
        query_filters.pop('group_id', None)
        query_filters.pop('domain_id', None)
        query_filters.pop('project_id', None)
        return self.build_role_assignment_query_url(effective=True, **query_filters)

    def _list_expected_role_assignments(self, **filters):
        """Given the filters, it returns expected direct role assignments.

        :param filters: filters that will be considered when listing role
                        assignments. Valid filters are: role_id, domain_id,
                        project_id, group_id, user_id and
                        inherited_to_projects.

        :returns: the list of the expected role assignments.

        """
        assignment_link = self.build_role_assignment_link(**filters)
        user_ids = [None]
        if filters.get('group_id'):
            user_ids = [user['id'] for user in PROVIDERS.identity_api.list_users_in_group(filters['group_id'])]
        else:
            user_ids = [self.default_user_id]
        project_ids = [None]
        if filters.get('domain_id'):
            project_ids = [project['id'] for project in PROVIDERS.resource_api.list_projects_in_domain(filters.pop('domain_id'))]
        else:
            project_ids = [project['id'] for project in PROVIDERS.resource_api.list_projects_in_subtree(self.project_id)]
        assignments = []
        for project_id in project_ids:
            filters['project_id'] = project_id
            for user_id in user_ids:
                filters['user_id'] = user_id
                assignments.append(self.build_role_assignment_entity(link=assignment_link, **filters))
        return assignments