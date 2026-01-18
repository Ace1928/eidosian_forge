from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_affinity_group_mappings(module):
    affinityGroupMappings = list()
    for affinityGroupMapping in module.params['affinity_group_mappings']:
        affinityGroupMappings.append(otypes.RegistrationAffinityGroupMapping(from_=otypes.AffinityGroup(name=affinityGroupMapping['source_name']) if affinityGroupMapping['source_name'] else None, to=otypes.AffinityGroup(name=affinityGroupMapping['dest_name']) if affinityGroupMapping['dest_name'] else None))
    return affinityGroupMappings