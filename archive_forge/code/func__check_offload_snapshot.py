from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from datetime import datetime
def _check_offload_snapshot(module, array):
    """Return Remote Snapshot (active or deleted) or None"""
    source_array = list(array.get_arrays().items)[0].name
    snapname = source_array + ':' + module.params['name'] + '.' + module.params['suffix']
    if _check_offload(module, array):
        res = array.get_remote_volume_snapshots(on=module.params['offload'], names=[snapname], destroyed=False)
    else:
        res = array.get_volume_snapshots(names=[snapname], destroyed=False)
    if res.status_code != 200:
        return None
    return list(res.items)[0]