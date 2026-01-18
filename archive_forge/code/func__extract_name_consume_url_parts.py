from openstack import exceptions
from openstack.orchestration.util import template_utils
from openstack.orchestration.v1 import resource as _resource
from openstack.orchestration.v1 import software_config as _sc
from openstack.orchestration.v1 import software_deployment as _sd
from openstack.orchestration.v1 import stack as _stack
from openstack.orchestration.v1 import stack_environment as _stack_environment
from openstack.orchestration.v1 import stack_event as _stack_event
from openstack.orchestration.v1 import stack_files as _stack_files
from openstack.orchestration.v1 import stack_template as _stack_template
from openstack.orchestration.v1 import template as _template
from openstack import proxy
from openstack import resource
def _extract_name_consume_url_parts(self, url_parts):
    if len(url_parts) == 3 and url_parts[0] == 'software_deployments' and (url_parts[1] == 'metadata'):
        return ['software_deployment', 'metadata']
    if url_parts[0] == 'stacks' and len(url_parts) > 2 and (not url_parts[2] in ['preview', 'resources']):
        del url_parts[2]
    return super(Proxy, self)._extract_name_consume_url_parts(url_parts)