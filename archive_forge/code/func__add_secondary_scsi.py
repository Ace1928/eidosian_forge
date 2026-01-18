from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _add_secondary_scsi(f, dc_name, attached, targets):
    f.write('  # Fill in the empty properties related to the secondary site\n')
    f.write('  dr_secondary_name: # %s\n' % attached.name)
    f.write('  dr_secondary_master_domain: # %s\n' % attached.master)
    f.write('  dr_secondary_dc_name: # %s\n' % dc_name)
    f.write('  dr_secondary_address: # %s\n' % attached.storage.volume_group.logical_units[0].address)
    f.write('  dr_secondary_port: # %s\n' % attached.storage.volume_group.logical_units[0].port)
    f.write('  # target example: ["target1","target2","target3"]\n')
    f.write('  dr_secondary_target: # [%s]\n' % ','.join(['"' + target + '"' for target in targets]))