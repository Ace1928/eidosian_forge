from keystone.common import cache
from keystone.common import manager
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.models import revoke_model
from keystone import notifications
def _consumer_callback(self, service, resource_type, operation, payload):
    self.revoke(revoke_model.RevokeEvent(consumer_id=payload['resource_info']))