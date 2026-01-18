from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
from re import sub
def _meraki_run_func_lookup(state):
    """Return the function that `meraki_run` will use based on `state`."""
    return {'query': _meraki_run_query, 'present': _meraki_run_present}[state]