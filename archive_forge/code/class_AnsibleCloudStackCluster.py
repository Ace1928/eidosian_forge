from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackCluster(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackCluster, self).__init__(module)
        self.returns = {'allocationstate': 'allocation_state', 'hypervisortype': 'hypervisor', 'clustertype': 'cluster_type', 'podname': 'pod', 'managedstate': 'managed_state', 'memoryovercommitratio': 'memory_overcommit_ratio', 'cpuovercommitratio': 'cpu_overcommit_ratio', 'ovm3vip': 'ovm3_vip'}
        self.cluster = None

    def _get_common_cluster_args(self):
        args = {'clustername': self.module.params.get('name'), 'hypervisor': self.module.params.get('hypervisor'), 'clustertype': self.module.params.get('cluster_type')}
        state = self.module.params.get('state')
        if state in ['enabled', 'disabled']:
            args['allocationstate'] = state.capitalize()
        return args

    def get_cluster(self):
        if not self.cluster:
            args = {}
            uuid = self.module.params.get('id')
            if uuid:
                args['id'] = uuid
                clusters = self.query_api('listClusters', **args)
                if clusters:
                    self.cluster = clusters['cluster'][0]
                    return self.cluster
            args['name'] = self.module.params.get('name')
            clusters = self.query_api('listClusters', **args)
            if clusters:
                self.cluster = clusters['cluster'][0]
                self.cluster['hypervisor'] = self.cluster['hypervisortype']
                self.cluster['clustername'] = self.cluster['name']
        return self.cluster

    def present_cluster(self):
        cluster = self.get_cluster()
        if cluster:
            cluster = self._update_cluster()
        else:
            cluster = self._create_cluster()
        return cluster

    def _create_cluster(self):
        required_params = ['cluster_type', 'hypervisor']
        self.module.fail_on_missing_params(required_params=required_params)
        args = self._get_common_cluster_args()
        args['zoneid'] = self.get_zone(key='id')
        args['podid'] = self.get_pod(key='id')
        args['url'] = self.module.params.get('url')
        args['username'] = self.module.params.get('username')
        args['password'] = self.module.params.get('password')
        args['guestvswitchname'] = self.module.params.get('guest_vswitch_name')
        args['guestvswitchtype'] = self.module.params.get('guest_vswitch_type')
        args['publicvswitchtype'] = self.module.params.get('public_vswitch_name')
        args['publicvswitchtype'] = self.module.params.get('public_vswitch_type')
        args['vsmipaddress'] = self.module.params.get('vms_ip_address')
        args['vsmusername'] = self.module.params.get('vms_username')
        args['vmspassword'] = self.module.params.get('vms_password')
        args['ovm3cluster'] = self.module.params.get('ovm3_cluster')
        args['ovm3pool'] = self.module.params.get('ovm3_pool')
        args['ovm3vip'] = self.module.params.get('ovm3_vip')
        self.result['changed'] = True
        cluster = None
        if not self.module.check_mode:
            res = self.query_api('addCluster', **args)
            if isinstance(res['cluster'], list):
                cluster = res['cluster'][0]
            else:
                cluster = res['cluster']
        return cluster

    def _update_cluster(self):
        cluster = self.get_cluster()
        args = self._get_common_cluster_args()
        args['id'] = cluster['id']
        if self.has_changed(args, cluster):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('updateCluster', **args)
                cluster = res['cluster']
        return cluster

    def absent_cluster(self):
        cluster = self.get_cluster()
        if cluster:
            self.result['changed'] = True
            args = {'id': cluster['id']}
            if not self.module.check_mode:
                self.query_api('deleteCluster', **args)
        return cluster