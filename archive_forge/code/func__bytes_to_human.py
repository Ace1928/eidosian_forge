from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def _bytes_to_human(bytes_number):
    if bytes_number:
        labels = ['B/s', 'KB/s', 'MB/s', 'GB/s', 'TB/s', 'PB/s']
        i = 0
        double_bytes = bytes_number
        while i < len(labels) and bytes_number >= 1024:
            double_bytes = bytes_number / 1024.0
            i += 1
            bytes_number = bytes_number / 1024
        return str(round(double_bytes, 2)) + ' ' + labels[i]
    return None