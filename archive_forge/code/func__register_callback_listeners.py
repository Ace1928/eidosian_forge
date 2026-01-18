from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _register_callback_listeners(self):
    notifications.register_event_callback(notifications.ACTIONS.deleted, 'user', self._delete_app_creds_on_user_delete_callback)
    notifications.register_event_callback(notifications.ACTIONS.disabled, 'user', self._delete_app_creds_on_user_delete_callback)
    notifications.register_event_callback(notifications.ACTIONS.internal, notifications.REMOVE_APP_CREDS_FOR_USER, self._delete_app_creds_on_assignment_removal)