import functools
from keystoneauth1 import access
from keystoneauth1.identity import access as access_plugin
from keystoneauth1.identity import generic
from keystoneauth1 import loading as ks_loading
from keystoneauth1 import session
from keystoneauth1 import token_endpoint
from oslo_config import cfg
from oslo_context import context
from oslo_db.sqlalchemy import enginefacade
from oslo_log import log as logging
import oslo_messaging
from oslo_utils import importutils
from heat.common import config
from heat.common import endpoint_utils
from heat.common import exception
from heat.common import policy
from heat.common import wsgi
from heat.engine import clients
def _create_auth_plugin(self):
    if self.auth_token_info:
        access_info = access.create(body=self.auth_token_info, auth_token=self.auth_token)
        return access_plugin.AccessInfoPlugin(auth_ref=access_info, auth_url=self.keystone_v3_endpoint)
    if self.password:
        return generic.Password(username=self.username, password=self.password, project_id=self.project_id, user_domain_id=self.user_domain_id, auth_url=self.keystone_v3_endpoint)
    if self.auth_token:
        return token_endpoint.Token(endpoint=self.keystone_v3_endpoint, token=self.auth_token)
    LOG.error('Keystone API connection failed, no password trust or auth_token!')
    raise exception.AuthorizationFailure()