from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _info_cmd_helper(self, cmd, node=None, delimiter=';'):
    """delimiter is for separate stats that come back, NOT for kv
        separation which is ="""
    if node is None:
        node = self._nodes[0]
    data = self._client.info_node(cmd, node)
    data = data.split('\t')
    if len(data) != 1 and len(data) != 2:
        self.module.fail_json(msg='Unexpected number of values returned in info command: ' + str(len(data)))
    data = data[-1]
    data = data.rstrip('\n\r')
    data_arr = data.split(delimiter)
    if '=' in data:
        retval = dict((metric.split('=', 1) for metric in data_arr))
    elif len(data_arr) == 1:
        retval = data_arr[0]
    else:
        retval = data_arr
    return retval