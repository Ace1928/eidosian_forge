import eventlet.queue
import functools
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common import exception
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import sync_point
from heat.objects import resource as resource_objects
from heat.rpc import api as rpc_api
from heat.rpc import listener_client
def _handle_resource_failure(self, cnxt, is_update, rsrc_id, stack, failure_reason):
    failure_handled = stack.mark_failed(failure_reason)
    if not failure_handled:
        self._retrigger_new_traversal(cnxt, stack.current_traversal, is_update, stack.id, rsrc_id)