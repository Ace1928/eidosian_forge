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
def _do_check_resource(self, cnxt, current_traversal, tmpl, resource_data, is_update, rsrc, stack, adopt_stack_data):
    prev_template_id = rsrc.current_template_id
    try:
        if is_update:
            requires = set((d.primary_key for d in resource_data.values() if d is not None))
            try:
                check_resource_update(rsrc, tmpl.id, requires, self.engine_id, stack, self.msg_queue)
            except resource.UpdateReplace:
                self._handle_resource_replacement(cnxt, current_traversal, tmpl.id, requires, rsrc, stack, adopt_stack_data)
                return False
        else:
            check_resource_cleanup(rsrc, tmpl.id, self.engine_id, stack.time_remaining(), self.msg_queue)
        return True
    except exception.UpdateInProgress:
        LOG.debug('Waiting for existing update to unlock resource %s', rsrc.id)
        if self._stale_resource_needs_retry(cnxt, rsrc, prev_template_id):
            rpc_data = sync_point.serialize_input_data(self.input_data)
            self._rpc_client.check_resource(cnxt, rsrc.id, current_traversal, rpc_data, is_update, adopt_stack_data)
        else:
            rsrc.handle_preempt()
    except exception.ResourceFailure as ex:
        action = ex.action or rsrc.action
        reason = 'Resource %s failed: %s' % (action, str(ex))
        self._handle_resource_failure(cnxt, is_update, rsrc.id, stack, reason)
    except scheduler.Timeout:
        self._handle_resource_failure(cnxt, is_update, rsrc.id, stack, u'Timed out')
    except CancelOperation:
        self._retrigger_new_traversal(cnxt, current_traversal, is_update, stack.id, rsrc.id)
    return False