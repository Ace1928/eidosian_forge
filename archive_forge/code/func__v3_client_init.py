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
def _v3_client_init(self):
    client = kc_v3.Client(session=self.session, connect_retries=cfg.CONF.client_retry_limit, interface=self._interface, region_name=self.auth_region_name)
    if hasattr(self.context.auth_plugin, 'get_access'):
        try:
            auth_ref = self.context.auth_plugin.get_access(self.session)
        except ks_exception.Unauthorized:
            LOG.error('Keystone client authentication failed')
            raise exception.AuthorizationFailure()
        if self.context.trust_id:
            if not auth_ref.trust_scoped:
                LOG.error('trust token re-scoping failed!')
                raise exception.AuthorizationFailure()
            if self.context.trustor_user_id != auth_ref.user_id:
                LOG.error('Trust impersonation failed')
                raise exception.AuthorizationFailure()
    return client