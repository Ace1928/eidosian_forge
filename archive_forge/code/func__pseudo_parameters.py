from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import parameters
def _pseudo_parameters(self, stack_identifier):
    stack_id = getattr(stack_identifier, 'stack_id', '')
    stack_name = getattr(stack_identifier, 'stack_name', '')
    tenant = getattr(stack_identifier, 'tenant', '')
    yield parameters.Parameter(self.PARAM_STACK_ID, parameters.Schema(parameters.Schema.STRING, _('Stack ID'), default=str(stack_id)))
    yield parameters.Parameter(self.PARAM_PROJECT_ID, parameters.Schema(parameters.Schema.STRING, _('Project ID'), default=str(tenant)))
    if stack_name:
        yield parameters.Parameter(self.PARAM_STACK_NAME, parameters.Schema(parameters.Schema.STRING, _('Stack Name'), default=stack_name))