import flask_restful
import http.client
from keystone.api._shared import json_home_relations
from keystone.api import endpoints as _endpoints_api
from keystone.catalog import schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
class EPFilterGroupsProjectsResource(ks_flask.ResourceBase):
    collection_key = 'project_endpoint_groups'
    member_key = 'project_endpoint_group'

    @classmethod
    def _add_self_referential_link(cls, ref, collection_name=None):
        url = '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s/projects/%(project_id)s' % {'endpoint_group_id': ref['endpoint_group_id'], 'project_id': ref['project_id']}
        ref.setdefault('links', {})
        ref['links']['self'] = url

    def get(self, endpoint_group_id, project_id):
        ENFORCER.enforce_call(action='identity:get_endpoint_group_in_project')
        PROVIDERS.resource_api.get_project(project_id)
        PROVIDERS.catalog_api.get_endpoint_group(endpoint_group_id)
        ref = PROVIDERS.catalog_api.get_endpoint_group_in_project(endpoint_group_id, project_id)
        return self.wrap_member(ref)

    def put(self, endpoint_group_id, project_id):
        ENFORCER.enforce_call(action='identity:add_endpoint_group_to_project')
        PROVIDERS.resource_api.get_project(project_id)
        PROVIDERS.catalog_api.get_endpoint_group(endpoint_group_id)
        PROVIDERS.catalog_api.add_endpoint_group_to_project(endpoint_group_id, project_id)
        return (None, http.client.NO_CONTENT)

    def delete(self, endpoint_group_id, project_id):
        ENFORCER.enforce_call(action='identity:remove_endpoint_group_from_project')
        PROVIDERS.resource_api.get_project(project_id)
        PROVIDERS.catalog_api.get_endpoint_group(endpoint_group_id)
        PROVIDERS.catalog_api.remove_endpoint_group_from_project(endpoint_group_id, project_id)
        return (None, http.client.NO_CONTENT)