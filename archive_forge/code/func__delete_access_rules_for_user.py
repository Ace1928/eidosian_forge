from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _delete_access_rules_for_user(self, user_id, initiator=None):
    """Delete all access rules for a user.

        :param str user_id: User ID

        This is triggered when a user is deleted.
        """
    access_rules = self.driver.list_access_rules_for_user(user_id, driver_hints.Hints())
    self.driver.delete_access_rules_for_user(user_id)
    for rule in access_rules:
        self.get_access_rule.invalidate(self, rule['id'])
        notifications.Audit.deleted(self._ACCESS_RULE, rule['id'], initiator)