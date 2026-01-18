from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _convert_to_minutes(hour):
    if hour[-2:] == 'AM' and hour[:2] == '12':
        return 0
    elif hour[-2:] == 'AM':
        return int(hour[:-2]) * 3600
    elif hour[-2:] == 'PM' and hour[:2] == '12':
        return 43200
    return (int(hour[:-2]) + 12) * 3600