import eventlet.queue
import functools
from oslo_log import log as logging
import oslo_messaging
from oslo_utils import excutils
from oslo_utils import uuidutils
from osprofiler import profiler
from heat.common import context
from heat.common import messaging as rpc_messaging
from heat.db import api as db_api
from heat.engine import check_resource
from heat.engine import node_data
from heat.engine import stack as parser
from heat.engine import sync_point
from heat.objects import stack as stack_objects
from heat.rpc import api as rpc_api
from heat.rpc import worker_client as rpc_client
def _stop_traversal(stack):
    old_trvsl = stack.current_traversal
    updated = _update_current_traversal(stack)
    if not updated:
        LOG.warning('Failed to update stack %(name)s with new traversal, aborting stack cancel', stack.name)
        return
    reason = 'Stack %(action)s cancelled' % {'action': stack.action}
    updated = stack.state_set(stack.action, stack.FAILED, reason)
    if not updated:
        LOG.warning('Failed to update stack %(name)s status to %(action)s_%(state)s', {'name': stack.name, 'action': stack.action, 'state': stack.FAILED})
        return
    sync_point.delete_all(stack.context, stack.id, old_trvsl)