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
@profiler.trace_cls('rpc')
class WorkerService(object):
    """Service that has 'worker' actor in convergence.

    This service is dedicated to handle internal messages to the 'worker'
    (a.k.a. 'converger') actor in convergence. Messages on this bus will
    use the 'cast' rather than 'call' method to anycast the message to
    an engine that will handle it asynchronously. It won't wait for
    or expect replies from these messages.
    """
    RPC_API_VERSION = '1.4'

    def __init__(self, host, topic, engine_id, thread_group_mgr):
        self.host = host
        self.topic = topic
        self.engine_id = engine_id
        self.thread_group_mgr = thread_group_mgr
        self._rpc_client = rpc_client.WorkerClient()
        self._rpc_server = None
        self.target = None

    def start(self):
        target = oslo_messaging.Target(version=self.RPC_API_VERSION, server=self.engine_id, topic=self.topic)
        self.target = target
        LOG.info('Starting %(topic)s (%(version)s) in engine %(engine)s.', {'topic': self.topic, 'version': self.RPC_API_VERSION, 'engine': self.engine_id})
        self._rpc_server = rpc_messaging.get_rpc_server(target, self)
        self._rpc_server.start()

    def stop(self):
        if self._rpc_server is None:
            return
        LOG.info('Stopping %(topic)s in engine %(engine)s.', {'topic': self.topic, 'engine': self.engine_id})
        try:
            self._rpc_server.stop()
            self._rpc_server.wait()
        except Exception as e:
            LOG.error('%(topic)s is failed to stop, %(exc)s', {'topic': self.topic, 'exc': e})

    def stop_traversal(self, stack):
        """Update current traversal to stop workers from propagating.

        Marks the stack as FAILED due to cancellation, but, allows all
        in_progress resources to complete normally; no worker is stopped
        abruptly.

        Any in-progress traversals are also stopped on all nested stacks that
        are descendants of the one passed.
        """
        _stop_traversal(stack)
        db_child_stacks = stack_objects.Stack.get_all_by_root_owner_id(stack.context, stack.id)
        for db_child in db_child_stacks:
            if db_child.status == parser.Stack.IN_PROGRESS:
                child = parser.Stack.load(stack.context, stack_id=db_child.id, stack=db_child, load_template=False)
                _stop_traversal(child)

    def stop_all_workers(self, stack):
        """Cancel all existing worker threads for the stack.

        Threads will stop running at their next yield point, whether or not the
        resource operations are complete.
        """
        cancelled = _cancel_workers(stack, self.thread_group_mgr, self.engine_id, self._rpc_client)
        if not cancelled:
            LOG.error('Failed to stop all workers of stack %s, stack cancel not complete', stack.name)
            return False
        LOG.info('[%(name)s(%(id)s)] Stopped all active workers for stack %(action)s', {'name': stack.name, 'id': stack.id, 'action': stack.action})
        return True

    def _retrigger_replaced(self, is_update, rsrc, stack, check_resource):
        graph = stack.convergence_dependencies.graph()
        key = parser.ConvergenceNode(rsrc.id, is_update)
        if key not in graph and rsrc.replaces is not None:
            values = {'action': rsrc.DELETE}
            db_api.resource_update_and_save(stack.context, rsrc.id, values)
            check_resource.retrigger_check_resource(stack.context, rsrc.replaces, stack)

    @context.request_context
    @log_exceptions
    def check_resource(self, cnxt, resource_id, current_traversal, data, is_update, adopt_stack_data, converge=False):
        """Process a node in the dependency graph.

        The node may be associated with either an update or a cleanup of its
        associated resource.
        """
        in_data = sync_point.deserialize_input_data(data)
        resource_data = node_data.load_resources_data(in_data if is_update else {})
        rsrc, stk_defn, stack = check_resource.load_resource(cnxt, resource_id, resource_data, current_traversal, is_update)
        if rsrc is None:
            return
        rsrc.converge = converge
        msg_queue = eventlet.queue.LightQueue()
        try:
            self.thread_group_mgr.add_msg_queue(stack.id, msg_queue)
            cr = check_resource.CheckResource(self.engine_id, self._rpc_client, self.thread_group_mgr, msg_queue, in_data)
            if current_traversal != stack.current_traversal:
                LOG.debug('[%s] Traversal cancelled; re-trigerring.', current_traversal)
                self._retrigger_replaced(is_update, rsrc, stack, cr)
            else:
                cr.check(cnxt, resource_id, current_traversal, resource_data, is_update, adopt_stack_data, rsrc, stack)
        finally:
            self.thread_group_mgr.remove_msg_queue(None, stack.id, msg_queue)

    @context.request_context
    @log_exceptions
    def cancel_check_resource(self, cnxt, stack_id):
        """Cancel check_resource for given stack.

        All the workers running for the given stack will be
        cancelled.
        """
        _cancel_check_resource(stack_id, self.engine_id, self.thread_group_mgr)