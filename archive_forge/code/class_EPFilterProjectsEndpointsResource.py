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
class EPFilterProjectsEndpointsResource(flask_restful.Resource):

    def get(self, project_id, endpoint_id):
        ENFORCER.enforce_call(action='identity:check_endpoint_in_project')
        PROVIDERS.catalog_api.get_endpoint(endpoint_id)
        PROVIDERS.resource_api.get_project(project_id)
        PROVIDERS.catalog_api.check_endpoint_in_project(endpoint_id, project_id)
        return (None, http.client.NO_CONTENT)

    def put(self, project_id, endpoint_id):
        ENFORCER.enforce_call(action='identity:add_endpoint_to_project')
        PROVIDERS.catalog_api.get_endpoint(endpoint_id)
        PROVIDERS.resource_api.get_project(project_id)
        PROVIDERS.catalog_api.add_endpoint_to_project(endpoint_id, project_id)
        return (None, http.client.NO_CONTENT)

    def delete(self, project_id, endpoint_id):
        ENFORCER.enforce_call(action='identity:remove_endpoint_from_project')
        return (PROVIDERS.catalog_api.remove_endpoint_from_project(endpoint_id, project_id), http.client.NO_CONTENT)