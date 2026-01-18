from __future__ import absolute_import, division, print_function
import datetime
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def _create_services_payload(self, service):
    if isinstance(service, list):
        return [{'id': s, 'type': 'service_reference'} for s in service]
    else:
        return [{'id': service, 'type': 'service_reference'}]