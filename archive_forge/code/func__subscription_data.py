from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from oslo_serialization import jsonutils
def _subscription_data(self):
    return {'subscriber': self._subscriber_url(), 'ttl': self.properties[self.TTL], 'options': self._subscription_options()}