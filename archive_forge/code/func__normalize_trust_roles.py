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
def _normalize_trust_roles(trust):
    trust_full_roles = []
    for trust_role in trust.get('roles', []):
        trust_role = trust_role['id']
        try:
            matching_role = PROVIDERS.role_api.get_role(trust_role)
            full_role = ks_flask.ResourceBase.wrap_member(matching_role, collection_name='roles', member_name='role')
            trust_full_roles.append(full_role['role'])
        except exception.RoleNotFound:
            pass
    trust['roles'] = trust_full_roles
    trust['roles_links'] = {'self': ks_flask.base_url(path='/%s/roles' % trust['id']), 'next': None, 'previous': None}