import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def _to_port(self, element):
    created = element.get('created_at')
    updated = element.get('updated_at')
    return OpenStack_2_PortInterface(id=element['id'], state=self.PORT_INTERFACE_MAP.get(element.get('status'), OpenStack_2_PortInterfaceState.UNKNOWN), created=created, driver=self, extra=dict(admin_state_up=element.get('admin_state_up'), allowed_address_pairs=element.get('allowed_address_pairs'), binding_vnic_type=element.get('binding:vnic_type'), binding_host_id=element.get('binding:host_id', None), device_id=element.get('device_id'), description=element.get('description', None), device_owner=element.get('device_owner'), fixed_ips=element.get('fixed_ips'), mac_address=element.get('mac_address'), name=element.get('name'), network_id=element.get('network_id'), project_id=element.get('project_id', None), port_security_enabled=element.get('port_security_enabled', None), revision_number=element.get('revision_number', None), security_groups=element.get('security_groups'), tags=element.get('tags', None), tenant_id=element.get('tenant_id'), updated=updated))