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
def _list_trusts(self):
    trustor_user_id = flask.request.args.get('trustor_user_id')
    trustee_user_id = flask.request.args.get('trustee_user_id')
    if trustor_user_id:
        target = {'trust': {'trustor_user_id': trustor_user_id}}
        ENFORCER.enforce_call(action='identity:list_trusts_for_trustor', target_attr=target)
    elif trustee_user_id:
        target = {'trust': {'trustee_user_id': trustee_user_id}}
        ENFORCER.enforce_call(action='identity:list_trusts_for_trustee', target_attr=target)
    else:
        ENFORCER.enforce_call(action='identity:list_trusts')
    trusts = []
    rules = policy._ENFORCER._enforcer.rules.get('identity:list_trusts')
    if isinstance(rules, op_checks.TrueCheck):
        LOG.warning('The policy check string for rule "identity:list_trusts" has been overridden to "always true". In the next release, this will cause the "identity:list_trusts" action to be fully permissive as hardcoded enforcement will be removed. To correct this issue, either stop overriding the "identity:list_trusts" rule in config to accept the defaults, or explicitly set a rule that is not empty.')
        if not flask.request.args:
            ENFORCER.enforce_call(action='admin_required')
    if not flask.request.args:
        trusts += PROVIDERS.trust_api.list_trusts()
    elif trustor_user_id:
        trusts += PROVIDERS.trust_api.list_trusts_for_trustor(trustor_user_id)
    elif trustee_user_id:
        trusts += PROVIDERS.trust_api.list_trusts_for_trustee(trustee_user_id)
    for trust in trusts:
        if 'roles' in trust:
            del trust['roles']
        if trust.get('expires_at') is not None:
            trust['expires_at'] = utils.isotime(trust['expires_at'], subsecond=True)
    return self.wrap_collection(trusts)