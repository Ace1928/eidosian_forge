import flask
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
class RoleAssignmentsAPI(ks_flask.APIBase):
    _name = 'role_assignments'
    _import_name = __name__
    resources = []
    resource_mapping = [ks_flask.construct_resource_map(resource=RoleAssignmentsResource, url='/role_assignments', resource_kwargs={}, rel='role_assignments')]