from oslo_log import log as logging
from heat.common import exception
from heat.engine import dependencies
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.objects import resource as resource_objects
def _update_resource_data(self, resource):
    node_data = resource.node_data(self.new_stack.defn)
    stk_defn.update_resource_data(self.existing_stack.defn, resource.name, node_data)
    stk_defn.update_resource_data(self.new_stack.defn, resource.name, node_data)