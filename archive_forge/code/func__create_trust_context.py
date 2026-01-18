import collections
import uuid
import weakref
from keystoneauth1 import exceptions as ks_exception
from keystoneauth1.identity import generic as ks_auth
from keystoneclient.v3 import client as kc_v3
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import importutils
from heat.common import config
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import password_gen
def _create_trust_context(self, trustor_user_id, trustor_proj_id):
    try:
        trustee_user_id = self.context.trusts_auth_plugin.get_user_id(self.session)
    except ks_exception.Unauthorized:
        LOG.error('Domain admin client authentication failed')
        raise exception.AuthorizationFailure()
    role_kw = {}
    if cfg.CONF.trusts_delegated_roles:
        role_kw['role_names'] = cfg.CONF.trusts_delegated_roles
    else:
        token_info = self.context.auth_token_info
        if token_info and token_info.get('token', {}).get('roles'):
            role_kw['role_ids'] = [r['id'] for r in token_info['token']['roles']]
        else:
            role_kw['role_names'] = self.context.roles
    allow_redelegation = cfg.CONF.reauthentication_auth_method == 'trusts' and cfg.CONF.allow_trusts_redelegation
    try:
        trust = self.client.trusts.create(trustor_user=trustor_user_id, trustee_user=trustee_user_id, project=trustor_proj_id, impersonation=True, allow_redelegation=allow_redelegation, **role_kw)
    except ks_exception.NotFound:
        LOG.debug('Failed to find roles %s for user %s' % (role_kw, trustor_user_id))
        raise exception.MissingCredentialError(required=_('roles %s') % role_kw)
    context_data = self.context.to_dict()
    context_data['overwrite'] = False
    trust_context = context.RequestContext.from_dict(context_data)
    trust_context.trust_id = trust.id
    trust_context.trustor_user_id = trustor_user_id
    return trust_context