from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
class VSANInfoManager(PyVmomi):

    def __init__(self, module):
        super(VSANInfoManager, self).__init__(module)
        self.datacenter = None
        self.cluster = None

    def gather_info(self):
        datacenter_name = self.module.params.get('datacenter')
        if datacenter_name:
            self.datacenter = self.find_datacenter_by_name(datacenter_name)
            if self.datacenter is None:
                self.module.fail_json(msg='Datacenter %s does not exist.' % datacenter_name)
        cluster_name = self.module.params.get('cluster_name')
        self.cluster = self.find_cluster_by_name(cluster_name=cluster_name, datacenter_name=self.datacenter)
        if self.cluster is None:
            self.module.fail_json(msg='Cluster %s does not exist.' % cluster_name)
        fetch_from_cache = self.module.params.get('fetch_from_cache')
        client_stub = self.si._GetStub()
        ssl_context = client_stub.schemeArgs.get('context')
        api_version = vsanapiutils.GetLatestVmodlVersion(self.module.params['hostname'])
        vc_mos = vsanapiutils.GetVsanVcMos(client_stub, context=ssl_context, version=api_version)
        vsan_cluster_health_system = vc_mos['vsan-cluster-health-system']
        cluster_health = {}
        try:
            cluster_health = vsan_cluster_health_system.VsanQueryVcClusterHealthSummary(cluster=self.cluster, fetchFromCache=fetch_from_cache)
        except vmodl.fault.NotFound as not_found:
            self.module.fail_json(msg=not_found.msg)
        except vmodl.fault.RuntimeFault as runtime_fault:
            self.module.fail_json(msg=runtime_fault.msg)
        health = json.dumps(cluster_health, cls=VmomiSupport.VmomiJSONEncoder, sort_keys=True, strip_dynamic=True)
        self.module.exit_json(changed=False, vsan_health_info=json.loads(health))