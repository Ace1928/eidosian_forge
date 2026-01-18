import json
import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import template_format
from heat.engine import attributes
from heat.engine import environment
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template
from heat.objects import raw_template
from heat.objects import stack as stack_object
from heat.objects import stack_lock
from heat.rpc import api as rpc_api
def _parse_nested_stack(self, stack_name, child_template, child_params, timeout_mins=None, adopt_data=None):
    if timeout_mins is None:
        timeout_mins = self.stack.timeout_mins
    stack_user_project_id = self.stack.stack_user_project_id
    new_nested_depth = self._child_nested_depth()
    child_env = environment.get_child_environment(self.stack.env, child_params, child_resource_name=self.name, item_to_remove=self.resource_info)
    parsed_template = self._child_parsed_template(child_template, child_env)
    self._validate_nested_resources(parsed_template)
    nested = parser.Stack(self.context, stack_name, parsed_template, timeout_mins=timeout_mins, disable_rollback=True, parent_resource=self.name, owner_id=self.stack.id, user_creds_id=self.stack.user_creds_id, stack_user_project_id=stack_user_project_id, adopt_stack_data=adopt_data, nested_depth=new_nested_depth)
    nested.set_parent_stack(self.stack)
    return nested