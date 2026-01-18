from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackPhysicalNetwork(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackPhysicalNetwork, self).__init__(module)
        self.returns = {'isolationmethods': 'isolation_method', 'broadcastdomainrange': 'broadcast_domain_range', 'networkspeed': 'network_speed', 'vlan': 'vlan', 'tags': 'tags'}
        self.nsps = []
        self.vrouters = None
        self.loadbalancers = None

    def _get_common_args(self):
        args = {'name': self.module.params.get('name'), 'isolationmethods': self.module.params.get('isolation_method'), 'broadcastdomainrange': self.module.params.get('broadcast_domain_range'), 'networkspeed': self.module.params.get('network_speed'), 'tags': self.module.params.get('tags'), 'vlan': self.module.params.get('vlan')}
        state = self.module.params.get('state')
        if state in ['enabled', 'disabled']:
            args['state'] = state.capitalize()
        return args

    def get_physical_network(self, key=None):
        physical_network = self.module.params.get('name')
        if self.physical_network:
            return self._get_by_key(key, self.physical_network)
        args = {'zoneid': self.get_zone(key='id')}
        physical_networks = self.query_api('listPhysicalNetworks', **args)
        if physical_networks:
            for net in physical_networks['physicalnetwork']:
                if physical_network.lower() in [net['name'].lower(), net['id']]:
                    self.physical_network = net
                    self.result['physical_network'] = net['name']
                    break
        return self._get_by_key(key, self.physical_network)

    def get_nsp(self, name=None):
        if not self.nsps:
            args = {'physicalnetworkid': self.get_physical_network(key='id')}
            res = self.query_api('listNetworkServiceProviders', **args)
            self.nsps = res['networkserviceprovider']
        names = []
        for nsp in self.nsps:
            names.append(nsp['name'])
            if nsp['name'].lower() == name.lower():
                return nsp
        self.module.fail_json(msg="Failed: '{0}' not in network service providers list '[{1}]'".format(name, names))

    def update_nsp(self, name=None, state=None, service_list=None):
        nsp = self.get_nsp(name)
        if not service_list and nsp['state'] == state:
            return nsp
        args = {'id': nsp['id'], 'servicelist': service_list, 'state': state}
        if not self.module.check_mode:
            res = self.query_api('updateNetworkServiceProvider', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                nsp = self.poll_job(res, 'networkserviceprovider')
        self.result['changed'] = True
        return nsp

    def get_vrouter_element(self, nsp_name='virtualrouter'):
        nsp = self.get_nsp(nsp_name)
        nspid = nsp['id']
        if self.vrouters is None:
            self.vrouters = dict()
            res = self.query_api('listVirtualRouterElements')
            for vrouter in res['virtualrouterelement']:
                self.vrouters[vrouter['nspid']] = vrouter
        if nspid not in self.vrouters:
            self.module.fail_json(msg="Failed: No VirtualRouterElement found for nsp '%s'" % nsp_name)
        return self.vrouters[nspid]

    def get_loadbalancer_element(self, nsp_name='internallbvm'):
        nsp = self.get_nsp(nsp_name)
        nspid = nsp['id']
        if self.loadbalancers is None:
            self.loadbalancers = dict()
            res = self.query_api('listInternalLoadBalancerElements')
            for loadbalancer in res['internalloadbalancerelement']:
                self.loadbalancers[loadbalancer['nspid']] = loadbalancer
            if nspid not in self.loadbalancers:
                self.module.fail_json(msg="Failed: No Loadbalancer found for nsp '%s'" % nsp_name)
        return self.loadbalancers[nspid]

    def set_vrouter_element_state(self, enabled, nsp_name='virtualrouter'):
        vrouter = self.get_vrouter_element(nsp_name)
        if vrouter['enabled'] == enabled:
            return vrouter
        args = {'id': vrouter['id'], 'enabled': enabled}
        if not self.module.check_mode:
            res = self.query_api('configureVirtualRouterElement', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                vrouter = self.poll_job(res, 'virtualrouterelement')
        self.result['changed'] = True
        return vrouter

    def set_loadbalancer_element_state(self, enabled, nsp_name='internallbvm'):
        loadbalancer = self.get_loadbalancer_element(nsp_name=nsp_name)
        if loadbalancer['enabled'] == enabled:
            return loadbalancer
        args = {'id': loadbalancer['id'], 'enabled': enabled}
        if not self.module.check_mode:
            res = self.query_api('configureInternalLoadBalancerElement', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                loadbalancer = self.poll_job(res, 'internalloadbalancerelement')
        self.result['changed'] = True
        return loadbalancer

    def present_network(self):
        network = self.get_physical_network()
        if network:
            network = self._update_network()
        else:
            network = self._create_network()
        return network

    def _create_network(self):
        self.result['changed'] = True
        args = dict(zoneid=self.get_zone(key='id'))
        args.update(self._get_common_args())
        if self.get_domain(key='id'):
            args['domainid'] = self.get_domain(key='id')
        if not self.module.check_mode:
            resource = self.query_api('createPhysicalNetwork', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.network = self.poll_job(resource, 'physicalnetwork')
        return self.network

    def _update_network(self):
        network = self.get_physical_network()
        args = dict(id=network['id'])
        args.update(self._get_common_args())
        if self.has_changed(args, network):
            self.result['changed'] = True
            if not self.module.check_mode:
                resource = self.query_api('updatePhysicalNetwork', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.physical_network = self.poll_job(resource, 'physicalnetwork')
        return self.physical_network

    def absent_network(self):
        physical_network = self.get_physical_network()
        if physical_network:
            self.result['changed'] = True
            args = {'id': physical_network['id']}
            if not self.module.check_mode:
                resource = self.query_api('deletePhysicalNetwork', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(resource, 'success')
        return physical_network