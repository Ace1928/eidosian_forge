from keystone.common import cache
from keystone.common import manager
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.models import revoke_model
from keystone import notifications
@MEMOIZE
def _list_events(self, last_fetch):
    return self.driver.list_events(last_fetch)