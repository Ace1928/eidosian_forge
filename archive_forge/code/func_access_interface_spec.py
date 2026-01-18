from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.aci.plugins.module_utils.aci import ACIModule, aci_argument_spec, aci_annotation_spec, aci_owner_spec, destination_epg_spec
def access_interface_spec():
    return dict(pod=dict(type='int', required=True, aliases=['pod_id', 'pod_number']), node=dict(type='int', required=True, aliases=['node_id']), path=dict(type='str', required=True), mtu=dict(type='int'))