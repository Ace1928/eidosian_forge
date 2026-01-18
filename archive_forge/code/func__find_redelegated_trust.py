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
def _find_redelegated_trust(self):
    redelegated_trust = None
    if self.oslo_context.is_delegated_auth:
        src_trust_id = self.oslo_context.trust_id
        if not src_trust_id:
            action = _('Redelegation allowed for delegated by trust only')
            raise exception.ForbiddenAction(action=action)
        redelegated_trust = PROVIDERS.trust_api.get_trust(src_trust_id)
    return redelegated_trust