from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def add_broadcast_domain_ports_rest(self, ports):
    """
        Add broadcast domain ports in rest.
        :param: ports to be added or moved.
        """
    api = 'network/ethernet/ports'
    body = {'broadcast_domain': {'name': self.parameters['resource_name'], 'ipspace': {'name': self.parameters['ipspace']}}}
    for port in ports:
        dummy, error = rest_generic.patch_async(self.rest_api, api, port['uuid'], body)
        if error:
            self.module.fail_json(msg=error)