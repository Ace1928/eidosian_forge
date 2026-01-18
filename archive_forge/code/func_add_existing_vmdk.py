from __future__ import absolute_import, division, print_function
import re
import time
import string
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils._text import to_text, to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def add_existing_vmdk(self, vm_obj, expected_disk_spec, diskspec, scsi_ctl):
    """
        Adds vmdk file described by expected_disk_spec['filename'], retrieves the file
        information and adds the correct spec to self.configspec.deviceChange.
        """
    filename = expected_disk_spec['filename']
    if vm_obj and diskspec.device.backing.fileName != filename or vm_obj is None:
        diskspec.device.backing.fileName = filename
        diskspec.device.key = -1
        self.change_detected = True
        self.configspec.deviceChange.append(diskspec)