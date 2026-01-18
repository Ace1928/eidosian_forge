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
def _trustor_trustee_only(trust):
    user_id = flask.request.environ.get(context.REQUEST_CONTEXT_ENV).user_id
    if user_id not in [trust.get('trustee_user_id'), trust.get('trustor_user_id')]:
        raise exception.ForbiddenAction(action=_('Requested user has no relation to this trust'))