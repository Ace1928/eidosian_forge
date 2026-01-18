import logging
from oslo_utils import encodeutils
from glance_store import exceptions
from glance_store.i18n import _, _LI
class MultiTenantConnectionManager(SwiftConnectionManager):

    def __init__(self, store, store_location, context=None, allow_reauth=False):
        if context is None:
            reason = _('Multi-tenant Swift storage requires a user context.')
            raise exceptions.BadStoreConfiguration(store_name='swift', reason=reason)
        super(MultiTenantConnectionManager, self).__init__(store, store_location, context, allow_reauth)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._client and self.client.trust_id:
            LOG.info(_LI('Revoking trust %s'), self.client.trust_id)
            self.client.trusts.delete(self.client.trust_id)

    def _get_storage_url(self):
        return self.location.swift_url

    def _init_connection(self):
        if self.allow_reauth:
            try:
                return super(MultiTenantConnectionManager, self)._init_connection()
            except Exception as e:
                LOG.debug('Cannot initialize swift connection for multi-tenant store with trustee token: %s. Using user token for connection initialization.', e)
                self.allow_reauth = False
        return self.store.get_store_connection(self.context.auth_token, self.storage_url)