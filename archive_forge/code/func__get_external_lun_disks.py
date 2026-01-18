from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _get_external_lun_disks(connection):
    external_disks = []
    disks_service = connection.system_service().disks_service()
    disks_list = disks_service.list()
    for disk in disks_list:
        if otypes.DiskStorageType.LUN == disk.storage_type:
            external_disks.append(disk)
    return external_disks