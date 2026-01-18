import uuid
from oslo_log import log
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.server import backends
def _bootstrap_admin_user(self):
    try:
        user = PROVIDERS.identity_api.get_user_by_name(self.admin_username, self.default_domain_id)
        LOG.info('User %s already exists, skipping creation.', self.admin_username)
        update = {}
        enabled = user['enabled']
        if not enabled:
            update['enabled'] = True
        try:
            PROVIDERS.identity_api.driver.authenticate(user['id'], self.admin_password)
        except AssertionError:
            update['password'] = self.admin_password
        if update:
            user = PROVIDERS.identity_api.update_user(user['id'], update)
            LOG.info('Reset password for user %s.', self.admin_username)
            if not enabled and user['enabled']:
                LOG.info('Enabled user %s.', self.admin_username)
    except exception.UserNotFound:
        user = PROVIDERS.identity_api.create_user(user_ref={'name': self.admin_username, 'enabled': True, 'domain_id': self.default_domain_id, 'password': self.admin_password})
        LOG.info('Created user %s', self.admin_username)
    self.admin_user_id = user['id']