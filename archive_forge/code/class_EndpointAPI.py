import flask_restful
import http.client
from keystone.api._shared import json_home_relations
from keystone.catalog import schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import utils
from keystone.common import validation
from keystone import exception
from keystone import notifications
from keystone.server import flask as ks_flask
class EndpointAPI(ks_flask.APIBase):
    _name = 'endpoints'
    _import_name = __name__
    resources = [EndpointResource]
    resource_mapping = [ks_flask.construct_resource_map(resource=EndpointPolicyEndpointResource, url='/endpoints/<string:endpoint_id>/OS-ENDPOINT-POLICY/policy', resource_kwargs={}, rel='endpoint_policy', resource_relation_func=_resource_rel_func, path_vars={'endpoint_id': json_home.Parameters.ENDPOINT_ID})]