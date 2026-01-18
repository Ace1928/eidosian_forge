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
def _get_trustor_roles(self, trust):
    original_trust = trust.copy()
    while original_trust.get('redelegated_trust_id'):
        original_trust = PROVIDERS.trust_api.get_trust(original_trust['redelegated_trust_id'])
    if not trust.get('project_id') in [None, '']:
        PROVIDERS.resource_api.get_project(trust['project_id'])
        assignment_list = PROVIDERS.assignment_api.list_role_assignments(user_id=original_trust['trustor_user_id'], project_id=original_trust['project_id'], effective=True, strip_domain_roles=False)
        return list({x['role_id'] for x in assignment_list})
    else:
        return []