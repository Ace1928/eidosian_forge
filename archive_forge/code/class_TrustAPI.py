import flask
import flask_restful
import http.client
from oslo_log import log
from oslo_policy import _checks as op_checks
from keystone.api._shared import json_home_relations
from keystone.common import context
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
from keystone.common import validation
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
from keystone.trust import schema
class TrustAPI(ks_flask.APIBase):
    _name = 'trusts'
    _import_name = __name__
    resources = [TrustResource]
    resource_mapping = [ks_flask.construct_resource_map(resource=RolesForTrustListResource, url='/trusts/<string:trust_id>/roles', resource_kwargs={}, rel='trust_roles', path_vars={'trust_id': TRUST_ID_PARAMETER_RELATION}, resource_relation_func=_build_resource_relation), ks_flask.construct_resource_map(resource=RoleForTrustResource, url='/trusts/<string:trust_id>/roles/<string:role_id>', resource_kwargs={}, rel='trust_role', path_vars={'trust_id': TRUST_ID_PARAMETER_RELATION, 'role_id': json_home.Parameters.ROLE_ID}, resource_relation_func=_build_resource_relation)]
    _api_url_prefix = '/OS-TRUST'