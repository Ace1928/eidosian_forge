from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class NetworksSwitchStacksRoutingInterfacesDhcp(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(dhcpMode=params.get('dhcpMode'), dhcpRelayServerIps=params.get('dhcpRelayServerIps'), dhcpLeaseTime=params.get('dhcpLeaseTime'), dnsNameserversOption=params.get('dnsNameserversOption'), dnsCustomNameservers=params.get('dnsCustomNameservers'), bootOptionsEnabled=params.get('bootOptionsEnabled'), bootNextServer=params.get('bootNextServer'), bootFileName=params.get('bootFileName'), dhcpOptions=params.get('dhcpOptions'), reservedIpRanges=params.get('reservedIpRanges'), fixedIpAssignments=params.get('fixedIpAssignments'), network_id=params.get('networkId'), switch_stack_id=params.get('switchStackId'), interface_id=params.get('interfaceId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('switchStackId') is not None or self.new_object.get('switch_stack_id') is not None:
            new_object_params['switchStackId'] = self.new_object.get('switchStackId') or self.new_object.get('switch_stack_id')
        if self.new_object.get('interfaceId') is not None or self.new_object.get('interface_id') is not None:
            new_object_params['interfaceId'] = self.new_object.get('interfaceId') or self.new_object.get('interface_id')
        return new_object_params

    def update_all_params(self):
        new_object_params = {}
        if self.new_object.get('dhcpMode') is not None or self.new_object.get('dhcp_mode') is not None:
            new_object_params['dhcpMode'] = self.new_object.get('dhcpMode') or self.new_object.get('dhcp_mode')
        if self.new_object.get('dhcpRelayServerIps') is not None or self.new_object.get('dhcp_relay_server_ips') is not None:
            new_object_params['dhcpRelayServerIps'] = self.new_object.get('dhcpRelayServerIps') or self.new_object.get('dhcp_relay_server_ips')
        if self.new_object.get('dhcpLeaseTime') is not None or self.new_object.get('dhcp_lease_time') is not None:
            new_object_params['dhcpLeaseTime'] = self.new_object.get('dhcpLeaseTime') or self.new_object.get('dhcp_lease_time')
        if self.new_object.get('dnsNameserversOption') is not None or self.new_object.get('dns_nameservers_option') is not None:
            new_object_params['dnsNameserversOption'] = self.new_object.get('dnsNameserversOption') or self.new_object.get('dns_nameservers_option')
        if self.new_object.get('dnsCustomNameservers') is not None or self.new_object.get('dns_custom_nameservers') is not None:
            new_object_params['dnsCustomNameservers'] = self.new_object.get('dnsCustomNameservers') or self.new_object.get('dns_custom_nameservers')
        if self.new_object.get('bootOptionsEnabled') is not None or self.new_object.get('boot_options_enabled') is not None:
            new_object_params['bootOptionsEnabled'] = self.new_object.get('bootOptionsEnabled')
        if self.new_object.get('bootNextServer') is not None or self.new_object.get('boot_next_server') is not None:
            new_object_params['bootNextServer'] = self.new_object.get('bootNextServer') or self.new_object.get('boot_next_server')
        if self.new_object.get('bootFileName') is not None or self.new_object.get('boot_file_name') is not None:
            new_object_params['bootFileName'] = self.new_object.get('bootFileName') or self.new_object.get('boot_file_name')
        if self.new_object.get('dhcpOptions') is not None or self.new_object.get('dhcp_options') is not None:
            new_object_params['dhcpOptions'] = self.new_object.get('dhcpOptions') or self.new_object.get('dhcp_options')
        if self.new_object.get('reservedIpRanges') is not None or self.new_object.get('reserved_ip_ranges') is not None:
            new_object_params['reservedIpRanges'] = self.new_object.get('reservedIpRanges') or self.new_object.get('reserved_ip_ranges')
        if self.new_object.get('fixedIpAssignments') is not None or self.new_object.get('fixed_ip_assignments') is not None:
            new_object_params['fixedIpAssignments'] = self.new_object.get('fixedIpAssignments') or self.new_object.get('fixed_ip_assignments')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('switchStackId') is not None or self.new_object.get('switch_stack_id') is not None:
            new_object_params['switchStackId'] = self.new_object.get('switchStackId') or self.new_object.get('switch_stack_id')
        if self.new_object.get('interfaceId') is not None or self.new_object.get('interface_id') is not None:
            new_object_params['interfaceId'] = self.new_object.get('interfaceId') or self.new_object.get('interface_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='switch', function='getNetworkSwitchStackRoutingInterfaceDhcp', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'name', name)
            if result is None:
                result = items
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        return result

    def exists(self):
        prev_obj = None
        id_exists = False
        name_exists = False
        o_id = self.new_object.get('networkId') or self.new_object.get('network_id')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_name(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('dhcpMode', 'dhcpMode'), ('dhcpRelayServerIps', 'dhcpRelayServerIps'), ('dhcpLeaseTime', 'dhcpLeaseTime'), ('dnsNameserversOption', 'dnsNameserversOption'), ('dnsCustomNameservers', 'dnsCustomNameservers'), ('bootOptionsEnabled', 'bootOptionsEnabled'), ('bootNextServer', 'bootNextServer'), ('bootFileName', 'bootFileName'), ('dhcpOptions', 'dhcpOptions'), ('reservedIpRanges', 'reservedIpRanges'), ('fixedIpAssignments', 'fixedIpAssignments'), ('networkId', 'networkId'), ('switchStackId', 'switchStackId'), ('interfaceId', 'interfaceId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        result = self.meraki.exec_meraki(family='switch', function='updateNetworkSwitchStackRoutingInterfaceDhcp', params=self.update_all_params(), op_modifies=True)
        return result