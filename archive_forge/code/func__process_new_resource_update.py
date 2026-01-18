from oslo_log import log as logging
from heat.common import exception
from heat.engine import dependencies
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.objects import resource as resource_objects
def _process_new_resource_update(self, new_res):
    res_name = new_res.name
    if res_name in self.existing_stack:
        existing_res = self.existing_stack[res_name]
        is_substituted = existing_res.check_is_substituted(type(new_res))
        if type(existing_res) is type(new_res) or is_substituted:
            try:
                yield from self._update_in_place(existing_res, new_res, is_substituted)
            except resource.UpdateReplace:
                pass
            else:
                LOG.debug('Storing definition of updated Resource %s', res_name)
                self.previous_stack.t.add_resource(new_res.t)
                self.previous_stack.t.store(self.previous_stack.context)
                self.existing_stack.t.add_resource(new_res.t)
                self.existing_stack.t.store(self.existing_stack.context)
                LOG.info('Resource %(res_name)s for stack %(stack_name)s updated', {'res_name': res_name, 'stack_name': self.existing_stack.name})
                self._update_resource_data(existing_res)
                return
        else:
            self._check_replace_restricted(new_res)
    yield from self._create_resource(new_res)