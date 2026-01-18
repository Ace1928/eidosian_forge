import flask
import flask_restful
import http.client
from keystone.api._shared import implied_roles as shared
from keystone.assignment import schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone.server import flask as ks_flask
class RoleAPI(ks_flask.APIBase):
    _name = 'roles'
    _import_name = __name__
    resources = [RoleResource]
    resource_mapping = [ks_flask.construct_resource_map(resource=RoleImplicationListResource, url='/roles/<string:prior_role_id>/implies', resource_kwargs={}, rel='implied_roles', path_vars={'prior_role_id': json_home.Parameters.ROLE_ID}), ks_flask.construct_resource_map(resource=RoleImplicationResource, resource_kwargs={}, url='/roles/<string:prior_role_id>/implies/<string:implied_role_id>', rel='implied_role', path_vars={'prior_role_id': json_home.Parameters.ROLE_ID, 'implied_role_id': json_home.Parameters.ROLE_ID})]