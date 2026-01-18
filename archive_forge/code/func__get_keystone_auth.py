import calendar
import time
import urllib
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization
from cryptography import x509 as cryptography_x509
from keystoneauth1 import identity
from keystoneauth1 import loading
from keystoneauth1 import service_token
from keystoneauth1 import session
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from castellan.common import exception
from castellan.common.objects import key as key_base_class
from castellan.common.objects import opaque_data as op_data
from castellan.i18n import _
from castellan.key_manager import key_manager
from barbicanclient import client as barbican_client_import
from barbicanclient import exceptions as barbican_exceptions
from oslo_utils import timeutils
def _get_keystone_auth(self, context):
    if context.__class__.__name__ == 'KeystonePassword':
        auth = identity.Password(auth_url=context.auth_url, username=context.username, password=context.password, user_id=context.user_id, user_domain_id=context.user_domain_id, user_domain_name=context.user_domain_name, trust_id=context.trust_id, domain_id=context.domain_id, domain_name=context.domain_name, project_id=context.project_id, project_name=context.project_name, project_domain_id=context.project_domain_id, project_domain_name=context.project_domain_name, reauthenticate=context.reauthenticate)
    elif context.__class__.__name__ == 'KeystoneToken':
        auth = identity.Token(auth_url=context.auth_url, token=context.token, trust_id=context.trust_id, domain_id=context.domain_id, domain_name=context.domain_name, project_id=context.project_id, project_name=context.project_name, project_domain_id=context.project_domain_id, project_domain_name=context.project_domain_name, reauthenticate=context.reauthenticate)
    elif context.__class__.__name__ == 'RequestContext':
        if getattr(context, 'get_auth_plugin', None):
            auth = context.get_auth_plugin()
        else:
            auth = identity.Token(auth_url=self.conf.barbican.auth_endpoint, token=context.auth_token, project_id=context.project_id, project_name=context.project_name, project_domain_id=context.project_domain_id, project_domain_name=context.project_domain_name)
    else:
        msg = _('context must be of type KeystonePassword, KeystoneToken, or RequestContext.')
        LOG.error(msg)
        raise exception.Forbidden(reason=msg)
    if self.conf.barbican.send_service_user_token:
        service_auth = loading.load_auth_from_conf_options(self.conf, group=_BARBICAN_SERVICE_USER_OPT_GROUP)
        auth = service_token.ServiceTokenAuthWrapper(user_auth=auth, service_auth=service_auth)
    return auth