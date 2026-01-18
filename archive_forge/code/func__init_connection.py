import logging
from oslo_utils import encodeutils
from glance_store import exceptions
from glance_store.i18n import _, _LI
def _init_connection(self):
    if self.allow_reauth:
        try:
            return super(MultiTenantConnectionManager, self)._init_connection()
        except Exception as e:
            LOG.debug('Cannot initialize swift connection for multi-tenant store with trustee token: %s. Using user token for connection initialization.', e)
            self.allow_reauth = False
    return self.store.get_store_connection(self.context.auth_token, self.storage_url)