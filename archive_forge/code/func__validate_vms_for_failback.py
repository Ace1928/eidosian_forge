from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _validate_vms_for_failback(self, setup_conn, setup_type):
    vms_in_preview = []
    vms_delete_protected = []
    service_setup = setup_conn.system_service().vms_service()
    for vm in service_setup.list():
        vm_service = service_setup.vm_service(vm.id)
        if vm.delete_protected:
            vms_delete_protected.append(vm.name)
        snapshots_service = vm_service.snapshots_service()
        for snapshot in snapshots_service.list():
            if snapshot.snapshot_status == types.SnapshotStatus.IN_PREVIEW:
                vms_in_preview.append(vm.name)
    if len(vms_in_preview) > 0:
        print("%s%sFailback process does not support VMs in preview. The '%s' setup contains the following previewed vms: '%s'%s" % (FAIL, PREFIX, setup_type, vms_in_preview, END))
        return False
    if len(vms_delete_protected) > 0:
        print("%s%sFailback process does not support delete protected VMs. The '%s' setup contains the following vms: '%s'%s" % (FAIL, PREFIX, setup_type, vms_delete_protected, END))
        return False
    return True