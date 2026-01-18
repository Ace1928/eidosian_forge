import binascii
from castellan.common import exception
from castellan.common.objects import symmetric_key
from oslo_config import cfg
from oslo_log import log as logging
class MigrationKeyManager(type(key_mgr)):

    def __init__(self, configuration):
        self.fixed_key = configuration.key_manager.fixed_key
        self.fixed_key_id = '00000000-0000-0000-0000-000000000000'
        super(MigrationKeyManager, self).__init__(configuration)

    def get(self, context, managed_object_id):
        if managed_object_id == self.fixed_key_id:
            LOG.debug('Processing request for secret associated with fixed_key key ID')
            if context is None:
                raise exception.Forbidden()
            key_bytes = bytes(binascii.unhexlify(self.fixed_key))
            secret = symmetric_key.SymmetricKey('AES', len(key_bytes) * 8, key_bytes)
        else:
            secret = super(MigrationKeyManager, self).get(context, managed_object_id)
        return secret

    def delete(self, context, managed_object_id):
        if managed_object_id == self.fixed_key_id:
            LOG.debug('Not deleting key associated with fixed_key key ID')
            if context is None:
                raise exception.Forbidden()
        else:
            super(MigrationKeyManager, self).delete(context, managed_object_id)