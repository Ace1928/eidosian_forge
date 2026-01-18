from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _delete_application_credentials_for_user(self, user_id, initiator=None):
    """Delete all application credentials for a user.

        :param str user_id: User ID

        This is triggered when a user is deleted.
        """
    app_creds = self.driver.list_application_credentials_for_user(user_id, driver_hints.Hints())
    self.driver.delete_application_credentials_for_user(user_id)
    for app_cred in app_creds:
        self.get_application_credential.invalidate(self, app_cred['id'])
        notifications.Audit.deleted(self._APP_CRED, app_cred['id'], initiator)