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