from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from oslo_serialization import jsonutils
def _get_subscription(self):
    return self.client().subscription(self.properties[self.QUEUE_NAME], id=self.resource_id, auto_create=False)