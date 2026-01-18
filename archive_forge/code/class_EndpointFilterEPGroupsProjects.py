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
class EndpointFilterEPGroupsProjects(flask_restful.Resource):

    def get(self, endpoint_group_id):
        ENFORCER.enforce_call(action='identity:list_projects_associated_with_endpoint_group')
        endpoint_group_refs = PROVIDERS.catalog_api.list_projects_associated_with_endpoint_group(endpoint_group_id)
        projects = []
        for endpoint_group_ref in endpoint_group_refs:
            project = PROVIDERS.resource_api.get_project(endpoint_group_ref['project_id'])
            if project:
                projects.append(project)
        return ks_flask.ResourceBase.wrap_collection(projects, collection_name='projects')