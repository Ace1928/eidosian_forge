from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _write_attached_storage_domains(f, dc_service, dc):
    """
    Add all the attached storage domains to the var file
    """
    attached_sds_service = dc_service.storage_domains_service()
    attached_sds_list = attached_sds_service.list()
    for attached_sd in attached_sds_list:
        if attached_sd.name == 'hosted_storage':
            f.write('# Hosted storage should not be part of the recovery process! Comment it out.\n')
            f.write('#- dr_domain_type: %s\n' % attached_sd.storage.type)
            f.write('#  dr_primary_name: %s\n' % attached_sd.name)
            f.write('#  dr_primary_dc_name: %s\n\n' % dc.name)
            continue
        if attached_sd.type == otypes.StorageDomainType.EXPORT:
            f.write('# Export storage domain should not be part of the recovery process!\n')
            f.write('# Please note that a data center with an export storage domain might reflect on the failback process.\n')
            f.write('#- dr_domain_type: %s\n' % attached_sd.storage.type)
            f.write('#  dr_primary_name: %s\n' % attached_sd.name)
            f.write('#  dr_primary_dc_name: %s\n\n' % dc.name)
            continue
        f.write('- dr_domain_type: %s\n' % attached_sd.storage.type)
        f.write('  dr_wipe_after_delete: %s\n' % attached_sd.wipe_after_delete)
        f.write('  dr_backup: %s\n' % attached_sd.backup)
        f.write('  dr_critical_space_action_blocker: %s\n' % attached_sd.critical_space_action_blocker)
        f.write('  dr_storage_domain_type: %s\n' % attached_sd.type)
        f.write('  dr_warning_low_space: %s\n' % attached_sd.warning_low_space_indicator)
        f.write('  dr_primary_name: %s\n' % attached_sd.name)
        f.write('  dr_primary_master_domain: %s\n' % attached_sd.master)
        f.write('  dr_primary_dc_name: %s\n' % dc.name)
        is_fcp = attached_sd._storage.type == otypes.StorageType.FCP
        is_scsi = attached_sd.storage.type == otypes.StorageType.ISCSI
        if not is_fcp and (not is_scsi):
            f.write('  dr_primary_path: %s\n' % attached_sd.storage.path)
            f.write('  dr_primary_address: %s\n' % attached_sd.storage.address)
            if attached_sd._storage.type == otypes.StorageType.POSIXFS:
                f.write('  dr_primary_vfs_type: %s\n' % attached_sd.storage.vfs_type)
            _add_secondary_mount(f, dc.name, attached_sd)
        else:
            f.write('  dr_discard_after_delete: %s\n' % attached_sd.discard_after_delete)
            f.write('  dr_domain_id: %s\n' % attached_sd.id)
            if attached_sd._storage._type == otypes.StorageType.ISCSI:
                f.write('  dr_primary_address: %s\n' % attached_sd.storage.volume_group.logical_units[0].address)
                f.write('  dr_primary_port: %s\n' % attached_sd.storage.volume_group.logical_units[0].port)
                targets = set((lun_unit.target for lun_unit in attached_sd.storage.volume_group.logical_units))
                f.write('  dr_primary_target: [%s]\n' % ','.join(['"' + target + '"' for target in targets]))
                _add_secondary_scsi(f, dc.name, attached_sd, targets)
            else:
                _add_secondary_fcp(f, dc.name, attached_sd)
        f.write('\n')