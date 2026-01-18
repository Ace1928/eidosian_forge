import json
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
@MEMOIZE
def _list_credentials_for_user(self, user_id, type):
    """List credentials for a specific user."""
    return self.driver.list_credentials_for_user(user_id, type)