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
class EPFilterProjectEndpointsListResource(flask_restful.Resource):

    def get(self, project_id):
        ENFORCER.enforce_call(action='identity:list_endpoints_for_project')
        PROVIDERS.resource_api.get_project(project_id)
        filtered_endpoints = PROVIDERS.catalog_api.list_endpoints_for_project(project_id)
        return ks_flask.ResourceBase.wrap_collection([_filter_endpoint(v) for v in filtered_endpoints.values()], collection_name='endpoints')