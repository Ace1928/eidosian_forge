from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, json
from ansible_collections.cisco.meraki.plugins.module_utils.network.meraki.meraki import (
from re import sub
def _construct_payload(params):
    """Recursively convert key names.

    This function recursively updates all dict key names from the
    Ansible/Python-style "snake case" to the format the Meraki API expects
    ("camel case").
    """
    payload = {}
    for k, v in params.items():
        if isinstance(v, dict):
            v = _construct_payload(v)
        payload[convert_to_camel_case(k)] = v
    return payload