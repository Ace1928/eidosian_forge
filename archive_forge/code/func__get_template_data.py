from heat.common import exception
from heat.common.i18n import _
from heat.engine import status
from heat.engine import template
from heat.rpc import api as rpc_api
def _get_template_data(self):
    if self._identity is None:
        return None
    if self._template_data is None:
        self._template_data = self._rpc_client.get_template(self._context, self._identity)
    return self._template_data