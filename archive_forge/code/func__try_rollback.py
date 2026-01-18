import json
import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import template_format
from heat.engine import attributes
from heat.engine import environment
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template
from heat.objects import raw_template
from heat.objects import stack as stack_object
from heat.objects import stack_lock
from heat.rpc import api as rpc_api
def _try_rollback(self):
    stack_identity = self.nested_identifier()
    if stack_identity is None:
        return False
    try:
        self.rpc_client().stack_cancel_update(self.context, dict(stack_identity), cancel_with_rollback=True)
    except exception.NotSupported:
        return False
    try:
        data = stack_object.Stack.get_status(self.context, self.resource_id)
    except exception.NotFound:
        return False
    action, status, status_reason, updated_time = data
    return status == self.stack.IN_PROGRESS