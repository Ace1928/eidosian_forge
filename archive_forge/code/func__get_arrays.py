from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def _get_arrays(array):
    """Get Connected Arrays"""
    arrays = []
    array_details = array.list_array_connections()
    for arraycnt in range(0, len(array_details)):
        arrays.append(array_details[arraycnt]['array_name'])
    return arrays