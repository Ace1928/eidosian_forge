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
def _retrigger_new_traversal(self, cnxt, current_traversal, is_update, stack_id, rsrc_id):
    latest_stack = parser.Stack.load(cnxt, stack_id=stack_id, force_reload=True)
    if current_traversal != latest_stack.current_traversal:
        self.retrigger_check_resource(cnxt, rsrc_id, latest_stack)