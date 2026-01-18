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
class RoleForTrustResource(flask_restful.Resource):

    @property
    def oslo_context(self):
        return flask.request.environ.get(context.REQUEST_CONTEXT_ENV, None)

    def get(self, trust_id, role_id):
        """Get a role that has been assigned to a trust."""
        ENFORCER.enforce_call(action='identity:get_role_for_trust', build_target=_build_trust_target_enforcement)
        if self.oslo_context.is_admin:
            raise exception.ForbiddenAction(action=_('Requested user has no relation to this trust'))
        trust = PROVIDERS.trust_api.get_trust(trust_id)
        rules = policy._ENFORCER._enforcer.rules.get('identity:get_role_for_trust')
        if isinstance(rules, op_checks.TrueCheck):
            LOG.warning('The policy check string for rule "identity:get_role_for_trust" has been overridden to "always true". In the next release, this will cause the "identity:get_role_for_trust" action to be fully permissive as hardcoded enforcement will be removed. To correct this issue, either stop overriding the "identity:get_role_for_trust" rule in config to accept the defaults, or explicitly set a rule that is not empty.')
            _trustor_trustee_only(trust)
        if not any((role['id'] == role_id for role in trust['roles'])):
            raise exception.RoleNotFound(role_id=role_id)
        role = PROVIDERS.role_api.get_role(role_id)
        return ks_flask.ResourceBase.wrap_member(role, collection_name='roles', member_name='role')