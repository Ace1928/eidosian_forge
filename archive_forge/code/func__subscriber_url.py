from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from oslo_serialization import jsonutils
def _subscriber_url(self):
    mistral_client = self.client('mistral')
    manager = getattr(mistral_client.executions, 'client', mistral_client.executions)
    return 'trust+%s/executions' % manager.http_client.base_url