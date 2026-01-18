from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils import arguments, errors, utils
def _update_payload_with_metric_attributes(payload, metric_attributes):
    if not metric_attributes:
        return
    payload['metrics'] = arguments.get_spec_payload(metric_attributes, *metric_attributes.keys())