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
class EPFilterEndpointProjectsResource(flask_restful.Resource):

    def get(self, endpoint_id):
        """Return a list of projects associated with the endpoint."""
        ENFORCER.enforce_call(action='identity:list_projects_for_endpoint')
        PROVIDERS.catalog_api.get_endpoint(endpoint_id)
        refs = PROVIDERS.catalog_api.list_projects_for_endpoint(endpoint_id)
        projects = [PROVIDERS.resource_api.get_project(ref['project_id']) for ref in refs]
        return ks_flask.ResourceBase.wrap_collection(projects, collection_name='projects')