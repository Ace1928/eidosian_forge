from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def _get_ip_dict(self, ip):
    ip_dict = dict(name=ip.name, id=ip.id, public_ip=ip.ip_address, public_ip_allocation_method=str(ip.public_ip_allocation_method))
    if ip.dns_settings:
        ip_dict['dns_settings'] = {'domain_name_label': ip.dns_settings.domain_name_label, 'fqdn': ip.dns_settings.fqdn}
    return ip_dict