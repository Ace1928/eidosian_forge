from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils import arguments, errors, utils
def _update_payload_with_check_attributes(payload, check_attributes):
    if not check_attributes:
        return
    if check_attributes['status']:
        check_attributes['status'] = STATUS_MAP[check_attributes['status']]
    filtered_attributes = arguments.get_spec_payload(check_attributes, *check_attributes.keys())
    payload['check'].update(filtered_attributes)