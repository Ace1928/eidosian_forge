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
def _initiate_propagate_resource(self, cnxt, resource_id, current_traversal, is_update, rsrc, stack):
    deps = stack.convergence_dependencies
    graph = deps.graph()
    graph_key = parser.ConvergenceNode(resource_id, is_update)
    if graph_key not in graph and rsrc.replaces is not None:
        graph_key = parser.ConvergenceNode(rsrc.replaces, is_update)

    def _get_input_data(req_node, input_forward_data=None):
        if req_node.is_update:
            if input_forward_data is None:
                return rsrc.node_data().as_dict()
            else:
                return input_forward_data
        elif req_node.rsrc_id != graph_key.rsrc_id:
            return rsrc.replaced_by if rsrc.replaced_by is not None else resource_id
        return None
    try:
        input_forward_data = None
        for req_node in sorted(deps.required_by(graph_key), key=lambda n: n.is_update):
            input_data = _get_input_data(req_node, input_forward_data)
            if req_node.is_update:
                input_forward_data = input_data
            propagate_check_resource(cnxt, self._rpc_client, req_node.rsrc_id, current_traversal, set(graph[req_node]), graph_key, input_data, req_node.is_update, stack.adopt_stack_data)
        if is_update:
            if input_forward_data is None:
                rsrc.clear_stored_attributes()
            else:
                rsrc.store_attributes()
        check_stack_complete(cnxt, stack, current_traversal, graph_key.rsrc_id, deps, graph_key.is_update)
    except exception.EntityNotFound as e:
        if e.entity == 'Sync Point':
            stack = parser.Stack.load(cnxt, stack_id=rsrc.stack.id, force_reload=True)
            if current_traversal == stack.current_traversal:
                LOG.debug('[%s] Traversal sync point missing.', current_traversal)
                return
            self.retrigger_check_resource(cnxt, resource_id, stack)
        else:
            raise