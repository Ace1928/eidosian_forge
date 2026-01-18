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
def _check_unrestricted(self):
    if self.oslo_context.is_admin:
        return
    token = self.auth_context['token']
    if 'application_credential' in token.methods:
        if not token.application_credential['unrestricted']:
            action = _("Using method 'application_credential' is not allowed for managing trusts.")
            raise exception.ForbiddenAction(action=action)