from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _assert_limit_not_exceeded(self, user_id):
    user_limit = CONF.application_credential.user_limit
    if user_limit >= 0:
        app_cred_count = len(self.list_application_credentials(user_id))
        if app_cred_count >= user_limit:
            raise exception.ApplicationCredentialLimitExceeded(limit=user_limit)