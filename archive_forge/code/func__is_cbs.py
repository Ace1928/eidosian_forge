from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _is_cbs(array):
    """Is the selected array a Cloud Block Store"""
    model = list(array.get_hardware(filter="type='controller'").items)[0].model
    is_cbs = bool('CBS' in model)
    return is_cbs