from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _write_external_lun_disks(f, external_disks, host_storages):
    f.write('\n# Mapping for external LUN disks\n')
    f.write('dr_lun_mappings:')
    for disk in external_disks:
        disk_id = disk.lun_storage.logical_units[0].id
        f.write('\n- logical_unit_alias: %s\n' % disk.alias)
        f.write('  logical_unit_description: %s\n' % disk.description)
        f.write('  wipe_after_delete: %s\n' % disk.wipe_after_delete)
        f.write('  shareable: %s\n' % disk.shareable)
        f.write('  primary_logical_unit_id: %s\n' % disk_id)
        disk_storage_type = ''
        if host_storages.get(disk_id) is not None:
            disk_storage_type = host_storages.get(disk_id).type
            disk_storage = host_storages.get(disk_id).logical_units[0]
            f.write('  primary_storage_type: %s\n' % disk_storage_type)
            if disk_storage_type == otypes.StorageType.ISCSI:
                portal = ''
                if disk_storage.portal is not None:
                    splitted = disk_storage.portal.split(',')
                    if len(splitted) > 0:
                        portal = splitted[1]
                f.write('  primary_logical_unit_address: %s\n  primary_logical_unit_port: %s\n  primary_logical_unit_portal: "%s"\n  primary_logical_unit_target: %s\n' % (disk_storage.address, disk_storage.port, portal, disk_storage.target))
                if disk_storage.username is not None:
                    f.write('  primary_logical_unit_username: %s\n  primary_logical_unit_password: PLEASE_SET_PASSWORD_HERE\n' % disk_storage.username)
        f.write('  # Fill in the following properties of the external LUN disk in the secondary site\n')
        f.write('  secondary_storage_type: %s\n' % (disk_storage_type if disk_storage_type != '' else 'STORAGE TYPE COULD NOT BE FETCHED!'))
        f.write('  secondary_logical_unit_id: # %s\n' % disk_id)
        if disk_storage_type == otypes.StorageType.ISCSI:
            f.write('  secondary_logical_unit_address: # %s\n  secondary_logical_unit_port: # %s\n  secondary_logical_unit_portal: # "%s"\n  secondary_logical_unit_target: # %s\n' % (disk_storage.address, disk_storage.port, portal, disk_storage.target))
            if disk_storage.username is not None:
                f.write('  secondary_logical_unit_username: # %s\n  secondary_logical_unit_password:PLEASE_SET_PASSWORD_HERE\n' % disk_storage.username)