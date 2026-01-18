from __future__ import annotations
import ipaddress
from typing import Any
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.floating_ips import BoundFloatingIP
from ..module_utils.vendor.hcloud.load_balancers import BoundLoadBalancer
from ..module_utils.vendor.hcloud.primary_ips import BoundPrimaryIP
from ..module_utils.vendor.hcloud.servers import BoundServer
def _get_rdns(self):
    ip_address = self.module.params.get('ip_address')
    try:
        ip_address_obj = ipaddress.ip_address(ip_address)
    except ValueError:
        self.module.fail_json(msg=f'The given IP address is not valid: {ip_address}')
    if ip_address_obj.version == 4:
        if self.module.params.get('server'):
            if self.hcloud_resource.public_net.ipv4.ip == ip_address:
                self.hcloud_rdns = {'ip_address': self.hcloud_resource.public_net.ipv4.ip, 'dns_ptr': self.hcloud_resource.public_net.ipv4.dns_ptr}
            else:
                self.module.fail_json(msg='The selected server does not have this IP address')
        elif self.module.params.get('floating_ip'):
            if self.hcloud_resource.ip == ip_address:
                self.hcloud_rdns = {'ip_address': self.hcloud_resource.ip, 'dns_ptr': self.hcloud_resource.dns_ptr[0]['dns_ptr']}
            else:
                self.module.fail_json(msg='The selected Floating IP does not have this IP address')
        elif self.module.params.get('primary_ip'):
            if self.hcloud_resource.ip == ip_address:
                self.hcloud_rdns = {'ip_address': self.hcloud_resource.ip, 'dns_ptr': self.hcloud_resource.dns_ptr[0]['dns_ptr']}
            else:
                self.module.fail_json(msg='The selected Primary IP does not have this IP address')
        elif self.module.params.get('load_balancer'):
            if self.hcloud_resource.public_net.ipv4.ip == ip_address:
                self.hcloud_rdns = {'ip_address': self.hcloud_resource.public_net.ipv4.ip, 'dns_ptr': self.hcloud_resource.public_net.ipv4.dns_ptr}
            else:
                self.module.fail_json(msg='The selected Load Balancer does not have this IP address')
    elif ip_address_obj.version == 6:
        if self.module.params.get('server'):
            for ipv6_address_dns_ptr in self.hcloud_resource.public_net.ipv6.dns_ptr:
                if ipv6_address_dns_ptr['ip'] == ip_address:
                    self.hcloud_rdns = {'ip_address': ipv6_address_dns_ptr['ip'], 'dns_ptr': ipv6_address_dns_ptr['dns_ptr']}
        elif self.module.params.get('floating_ip'):
            for ipv6_address_dns_ptr in self.hcloud_resource.dns_ptr:
                if ipv6_address_dns_ptr['ip'] == ip_address:
                    self.hcloud_rdns = {'ip_address': ipv6_address_dns_ptr['ip'], 'dns_ptr': ipv6_address_dns_ptr['dns_ptr']}
        elif self.module.params.get('primary_ip'):
            for ipv6_address_dns_ptr in self.hcloud_resource.dns_ptr:
                if ipv6_address_dns_ptr['ip'] == ip_address:
                    self.hcloud_rdns = {'ip_address': ipv6_address_dns_ptr['ip'], 'dns_ptr': ipv6_address_dns_ptr['dns_ptr']}
        elif self.module.params.get('load_balancer'):
            for ipv6_address_dns_ptr in self.hcloud_resource.public_net.ipv6.dns_ptr:
                if ipv6_address_dns_ptr['ip'] == ip_address:
                    self.hcloud_rdns = {'ip_address': ipv6_address_dns_ptr['ip'], 'dns_ptr': ipv6_address_dns_ptr['dns_ptr']}