import ipaddress
import time
import warnings
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
from openstack import warnings as os_warnings
def _normalize_floating_ip(self, ip):
    location = self._get_current_location(project_id=ip.get('owner'))
    ip = ip.copy()
    ret = utils.Munch(location=location)
    fixed_ip_address = ip.pop('fixed_ip_address', ip.pop('fixed_ip', None))
    floating_ip_address = ip.pop('floating_ip_address', ip.pop('ip', None))
    network_id = ip.pop('floating_network_id', ip.pop('network', ip.pop('pool', None)))
    project_id = ip.pop('tenant_id', '')
    project_id = ip.pop('project_id', project_id)
    instance_id = ip.pop('instance_id', None)
    router_id = ip.pop('router_id', None)
    id = ip.pop('id')
    port_id = ip.pop('port_id', None)
    created_at = ip.pop('created_at', None)
    updated_at = ip.pop('updated_at', None)
    description = ip.pop('description', '')
    revision_number = ip.pop('revision_number', None)
    if self._use_neutron_floating():
        attached = bool(port_id)
        status = ip.pop('status', 'UNKNOWN')
    else:
        attached = bool(instance_id)
        status = 'ACTIVE'
    ret = utils.Munch(attached=attached, fixed_ip_address=fixed_ip_address, floating_ip_address=floating_ip_address, id=id, location=self._get_current_location(project_id=project_id), network=network_id, port=port_id, router=router_id, status=status, created_at=created_at, updated_at=updated_at, description=description, revision_number=revision_number, properties=ip.copy())
    if not self.strict_mode:
        ret['port_id'] = port_id
        ret['router_id'] = router_id
        ret['project_id'] = project_id
        ret['tenant_id'] = project_id
        ret['floating_network_id'] = network_id
        for key, val in ret['properties'].items():
            ret.setdefault(key, val)
    return ret