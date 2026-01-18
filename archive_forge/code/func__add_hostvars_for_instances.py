from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _add_hostvars_for_instances(self):
    """Add hostvars for instances in the dynamic inventory."""
    ip_style = self.get_option('ip_style')
    for instance in self.instances:
        hostvars = instance._raw_json
        hostname = make_unsafe(instance.label)
        for hostvar_key in hostvars:
            if ip_style == 'api' and hostvar_key in ['ipv4', 'ipv6']:
                continue
            self.inventory.set_variable(hostname, hostvar_key, make_unsafe(hostvars[hostvar_key]))
        if ip_style == 'api':
            ips = instance.ips.ipv4.public + instance.ips.ipv4.private
            ips += [instance.ips.ipv6.slaac, instance.ips.ipv6.link_local]
            ips += instance.ips.ipv6.pools
            for ip_type in set((ip.type for ip in ips)):
                self.inventory.set_variable(hostname, ip_type, make_unsafe(self._ip_data([ip for ip in ips if ip.type == ip_type])))