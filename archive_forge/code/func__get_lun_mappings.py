from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_lun_mappings(module):
    lunMappings = list()
    for lunMapping in module.params['lun_mappings']:
        (lunMappings.append(otypes.RegistrationLunMapping(from_=otypes.Disk(lun_storage=otypes.HostStorage(type=otypes.StorageType(lunMapping['source_storage_type']) if lunMapping['source_storage_type'] in ['iscsi', 'fcp'] else None, logical_units=[otypes.LogicalUnit(id=lunMapping['source_logical_unit_id'])])) if lunMapping['source_logical_unit_id'] else None, to=otypes.Disk(lun_storage=otypes.HostStorage(type=otypes.StorageType(lunMapping['dest_storage_type']) if lunMapping['dest_storage_type'] in ['iscsi', 'fcp'] else None, logical_units=[otypes.LogicalUnit(id=lunMapping.get('dest_logical_unit_id'), port=lunMapping.get('dest_logical_unit_port'), portal=lunMapping.get('dest_logical_unit_portal'), address=lunMapping.get('dest_logical_unit_address'), target=lunMapping.get('dest_logical_unit_target'), password=lunMapping.get('dest_logical_unit_password'), username=lunMapping.get('dest_logical_unit_username'))])) if lunMapping['dest_logical_unit_id'] else None)),)
    return lunMappings