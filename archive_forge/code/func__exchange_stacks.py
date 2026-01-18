from oslo_log import log as logging
from heat.common import exception
from heat.engine import dependencies
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.objects import resource as resource_objects
@staticmethod
def _exchange_stacks(existing_res, prev_res):
    resource_objects.Resource.exchange_stacks(existing_res.stack.context, existing_res.id, prev_res.id)
    prev_stack, existing_stack = (prev_res.stack, existing_res.stack)
    prev_stack.add_resource(existing_res)
    existing_stack.add_resource(prev_res)