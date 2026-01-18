from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
@staticmethod
def _parse_ips_aslist(ovirt_vms, version_condition=lambda version: True):
    ips = []
    for ovirt_vm in ovirt_vms:
        for device in ovirt_vm.get('reported_devices', []):
            for curr_ip in device.get('ips', []):
                if version_condition(curr_ip.get('version')):
                    ips.append(curr_ip.get('address'))
    return ips