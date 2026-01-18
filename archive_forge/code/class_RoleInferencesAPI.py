import flask_restful
from keystone.api._shared import implied_roles as shared
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.server import flask as ks_flask
class RoleInferencesAPI(ks_flask.APIBase):
    _name = 'role_inferences'
    _import_name = __name__
    resources = []
    resource_mapping = [ks_flask.construct_resource_map(resource=RoleInferencesResource, url='/role_inferences', resource_kwargs={}, rel='role_inferences')]