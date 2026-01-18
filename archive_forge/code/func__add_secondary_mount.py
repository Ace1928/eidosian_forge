from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _add_secondary_mount(f, dc_name, attached):
    f.write('  # Fill in the empty properties related to the secondary site\n')
    f.write('  dr_secondary_name: # %s\n' % attached.name)
    f.write('  dr_secondary_master_domain: # %s\n' % attached.master)
    f.write('  dr_secondary_dc_name: # %s\n' % dc_name)
    f.write('  dr_secondary_path: # %s\n' % attached.storage.path)
    f.write('  dr_secondary_address: # %s\n' % attached.storage.address)
    if attached._storage.type == otypes.StorageType.POSIXFS:
        f.write('  dr_secondary_vfs_type: # %s\n' % attached.storage.vfs_type)