from heat.common import exception
from heat.common.i18n import _
from heat.engine import status
from heat.engine import template
from heat.rpc import api as rpc_api
def _members(self, include_failed):
    return (r for r in self._get_member_data() if include_failed or r[rpc_api.RES_STATUS] != status.ResourceStatus.FAILED)