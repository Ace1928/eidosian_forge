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
def _retrigger_replaced(self, is_update, rsrc, stack, check_resource):
    graph = stack.convergence_dependencies.graph()
    key = parser.ConvergenceNode(rsrc.id, is_update)
    if key not in graph and rsrc.replaces is not None:
        values = {'action': rsrc.DELETE}
        db_api.resource_update_and_save(stack.context, rsrc.id, values)
        check_resource.retrigger_check_resource(stack.context, rsrc.replaces, stack)