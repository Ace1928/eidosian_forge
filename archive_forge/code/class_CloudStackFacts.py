from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.facts import ansible_collector, default_collectors
class CloudStackFacts(object):

    def __init__(self):
        collector = ansible_collector.get_ansible_collector(all_collector_classes=default_collectors.collectors, filter_spec='default_ipv4', gather_subset=['!all', 'network'], gather_timeout=10)
        self.facts = collector.collect(module)
        self.api_ip = None
        self.fact_paths = {'cloudstack_service_offering': 'service-offering', 'cloudstack_availability_zone': 'availability-zone', 'cloudstack_public_hostname': 'public-hostname', 'cloudstack_public_ipv4': 'public-ipv4', 'cloudstack_local_hostname': 'local-hostname', 'cloudstack_local_ipv4': 'local-ipv4', 'cloudstack_instance_id': 'instance-id'}

    def run(self):
        result = {}
        filter = module.params.get('filter')
        if not filter:
            for key, path in self.fact_paths.items():
                result[key] = self._fetch(CS_METADATA_BASE_URL + '/' + path)
            result['cloudstack_user_data'] = self._get_user_data_json()
        elif filter == 'cloudstack_user_data':
            result['cloudstack_user_data'] = self._get_user_data_json()
        elif filter in self.fact_paths:
            result[filter] = self._fetch(CS_METADATA_BASE_URL + '/' + self.fact_paths[filter])
        return result

    def _get_user_data_json(self):
        try:
            return yaml.safe_load(self._fetch(CS_USERDATA_BASE_URL))
        except Exception:
            return None

    def _fetch(self, path):
        api_ip = self._get_api_ip()
        if not api_ip:
            return None
        api_url = path % api_ip
        response, info = fetch_url(module, api_url, force=True)
        if response:
            data = response.read()
        else:
            data = None
        return data

    def _get_dhcp_lease_file(self):
        """Return the path of the lease file."""
        default_iface = self.facts['default_ipv4']['interface']
        dhcp_lease_file_locations = ['/var/lib/dhcp/dhclient.%s.leases' % default_iface, '/var/lib/dhclient/dhclient-%s.leases' % default_iface, '/var/lib/dhclient/dhclient--%s.lease' % default_iface, '/var/db/dhclient.leases.%s' % default_iface]
        for file_path in dhcp_lease_file_locations:
            if os.path.exists(file_path):
                return file_path
        module.fail_json(msg='Could not find dhclient leases file.')

    def _get_api_ip(self):
        """Return the IP of the DHCP server."""
        if module.params.get('meta_data_host'):
            return module.params.get('meta_data_host')
        elif not self.api_ip:
            dhcp_lease_file = self._get_dhcp_lease_file()
            for line in open(dhcp_lease_file):
                if 'dhcp-server-identifier' in line:
                    line = line.translate(None, ';')
                    self.api_ip = line.split()[2]
                    break
            if not self.api_ip:
                module.fail_json(msg='No dhcp-server-identifier found in leases file.')
        return self.api_ip