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
def _handle_resource_replacement(self, cnxt, current_traversal, new_tmpl_id, requires, rsrc, stack, adopt_stack_data):
    """Create a replacement resource and trigger a check on it."""
    try:
        new_res_id = rsrc.make_replacement(new_tmpl_id, requires)
    except exception.UpdateInProgress:
        LOG.info('No replacement created - resource already locked by new traversal')
        return
    if new_res_id is None:
        LOG.info('No replacement created - new traversal already in progress')
        self._retrigger_new_traversal(cnxt, current_traversal, True, stack.id, rsrc.id)
        return
    LOG.info('Replacing resource with new id %s', new_res_id)
    rpc_data = sync_point.serialize_input_data(self.input_data)
    self._rpc_client.check_resource(cnxt, new_res_id, current_traversal, rpc_data, True, adopt_stack_data)