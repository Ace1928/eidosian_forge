from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_vnic_profile_mappings(module):
    vnicProfileMappings = list()
    for vnicProfileMapping in module.params['vnic_profile_mappings']:
        vnicProfileMappings.append(otypes.VnicProfileMapping(source_network_name=vnicProfileMapping['source_network_name'], source_network_profile_name=vnicProfileMapping['source_profile_name'], target_vnic_profile=otypes.VnicProfile(id=vnicProfileMapping['target_profile_id']) if vnicProfileMapping['target_profile_id'] else None))
    return vnicProfileMappings