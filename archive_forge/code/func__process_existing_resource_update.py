from oslo_log import log as logging
from heat.common import exception
from heat.engine import dependencies
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.objects import resource as resource_objects
def _process_existing_resource_update(self, existing_res):
    res_name = existing_res.name
    if res_name in self.previous_stack:
        backup_res = self.previous_stack[res_name]
        yield from self._remove_backup_resource(backup_res)
    if res_name in self.new_stack:
        new_res = self.new_stack[res_name]
        if new_res.state == (new_res.INIT, new_res.COMPLETE):
            return
    if existing_res.stack is not self.previous_stack:
        yield from existing_res.destroy()
    if res_name not in self.new_stack:
        self.existing_stack.remove_resource(res_name)