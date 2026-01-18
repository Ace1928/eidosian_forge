from heat.common import exception
from heat.common.i18n import _
from heat.engine import status
from heat.engine import template
from heat.rpc import api as rpc_api
def _get_member_data(self):
    if self._identity is None:
        return []
    if self._member_data is None:
        rsrcs = self._rpc_client.list_stack_resources(self._context, dict(self._identity))

        def sort_key(r):
            return (r[rpc_api.RES_STATUS] != status.ResourceStatus.FAILED, r[rpc_api.RES_CREATION_TIME], r[rpc_api.RES_NAME])
        self._member_data = sorted(rsrcs, key=sort_key)
    return self._member_data