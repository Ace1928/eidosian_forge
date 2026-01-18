from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackNetwork(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackNetwork, self).__init__(module)
        self.returns = {'networkdomain': 'network_domain', 'networkofferingname': 'network_offering', 'networkofferingdisplaytext': 'network_offering_display_text', 'networkofferingconservemode': 'network_offering_conserve_mode', 'networkofferingavailability': 'network_offering_availability', 'aclid': 'acl_id', 'issystem': 'is_system', 'ispersistent': 'is_persistent', 'acltype': 'acl_type', 'type': 'type', 'traffictype': 'traffic_type', 'ip6gateway': 'gateway_ipv6', 'ip6cidr': 'cidr_ipv6', 'gateway': 'gateway', 'cidr': 'cidr', 'netmask': 'netmask', 'broadcastdomaintype': 'broadcast_domain_type', 'dns1': 'dns1', 'dns2': 'dns2'}
        self.network = None

    def get_network_acl(self, key=None, acl_id=None):
        if acl_id is not None:
            args = {'id': acl_id, 'vpcid': self.get_vpc(key='id')}
        else:
            acl_name = self.module.params.get('acl')
            if not acl_name:
                return
            args = {'name': acl_name, 'vpcid': self.get_vpc(key='id')}
        network_acls = self.query_api('listNetworkACLLists', **args)
        if network_acls:
            acl = network_acls['networkacllist'][0]
            return self._get_by_key(key, acl)

    def get_network_offering(self, key=None):
        network_offering = self.module.params.get('network_offering')
        if not network_offering:
            self.module.fail_json(msg='missing required arguments: network_offering')
        args = {'zoneid': self.get_zone(key='id'), 'fetch_list': True}
        network_offerings = self.query_api('listNetworkOfferings', **args)
        if network_offerings:
            for no in network_offerings:
                if network_offering in [no['name'], no['displaytext'], no['id']]:
                    return self._get_by_key(key, no)
        self.module.fail_json(msg="Network offering '%s' not found" % network_offering)

    def _get_args(self):
        args = {'name': self.module.params.get('name'), 'displaytext': self.get_or_fallback('display_text', 'name'), 'networkdomain': self.module.params.get('network_domain'), 'networkofferingid': self.get_network_offering(key='id')}
        return args

    def query_network(self, refresh=False):
        if not self.network or refresh:
            network = self.module.params.get('name')
            args = {'zoneid': self.get_zone(key='id'), 'projectid': self.get_project(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'vpcid': self.get_vpc(key='id'), 'fetch_list': True}
            networks = self.query_api('listNetworks', **args)
            if networks:
                for n in networks:
                    if network in [n['name'], n['displaytext'], n['id']]:
                        self.network = n
                        self.network['acl'] = self.get_network_acl(key='name', acl_id=n.get('aclid'))
                        break
        return self.network

    def present_network(self):
        if self.module.params.get('acl') is not None and self.module.params.get('vpc') is None:
            self.module.fail_json(msg='Missing required params: vpc')
        network = self.query_network()
        if not network:
            network = self.create_network(network)
        else:
            network = self.update_network(network)
        if network:
            network = self.ensure_tags(resource=network, resource_type='Network')
        return network

    def update_network(self, network):
        args = self._get_args()
        args['id'] = network['id']
        if self.has_changed(args, network):
            self.result['changed'] = True
            if not self.module.check_mode:
                network = self.query_api('updateNetwork', **args)
                poll_async = self.module.params.get('poll_async')
                if network and poll_async:
                    network = self.poll_job(network, 'network')
        if network.get('aclid') != self.get_network_acl(key='id'):
            self.result['changed'] = True
            if not self.module.check_mode:
                args = {'aclid': self.get_network_acl(key='id'), 'networkid': network['id']}
                network = self.query_api('replaceNetworkACLList', **args)
                if self.module.params.get('poll_async'):
                    self.poll_job(network, 'networkacllist')
                    network = self.query_network(refresh=True)
        return network

    def create_network(self, network):
        self.result['changed'] = True
        args = self._get_args()
        args.update({'acltype': self.module.params.get('acl_type'), 'aclid': self.get_network_acl(key='id'), 'zoneid': self.get_zone(key='id'), 'projectid': self.get_project(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'startip': self.module.params.get('start_ip'), 'endip': self.get_or_fallback('end_ip', 'start_ip'), 'netmask': self.module.params.get('netmask'), 'gateway': self.module.params.get('gateway'), 'startipv6': self.module.params.get('start_ipv6'), 'endipv6': self.get_or_fallback('end_ipv6', 'start_ipv6'), 'ip6cidr': self.module.params.get('cidr_ipv6'), 'ip6gateway': self.module.params.get('gateway_ipv6'), 'vlan': self.module.params.get('vlan'), 'isolatedpvlan': self.module.params.get('isolated_pvlan'), 'subdomainaccess': self.module.params.get('subdomain_access'), 'vpcid': self.get_vpc(key='id')})
        if not self.module.check_mode:
            res = self.query_api('createNetwork', **args)
            network = res['network']
        return network

    def restart_network(self):
        network = self.query_network()
        if not network:
            self.module.fail_json(msg="No network named '%s' found." % self.module.params('name'))
        if network['state'].lower() in ['implemented', 'setup']:
            self.result['changed'] = True
            args = {'id': network['id'], 'cleanup': self.module.params.get('clean_up')}
            if not self.module.check_mode:
                network = self.query_api('restartNetwork', **args)
                poll_async = self.module.params.get('poll_async')
                if network and poll_async:
                    network = self.poll_job(network, 'network')
        return network

    def absent_network(self):
        network = self.query_network()
        if network:
            self.result['changed'] = True
            args = {'id': network['id']}
            if not self.module.check_mode:
                res = self.query_api('deleteNetwork', **args)
                poll_async = self.module.params.get('poll_async')
                if res and poll_async:
                    self.poll_job(res, 'network')
            return network

    def get_result(self, resource):
        super(AnsibleCloudStackNetwork, self).get_result(resource)
        if resource:
            self.result['acl'] = self.get_network_acl(key='name', acl_id=resource.get('aclid'))
        return self.result