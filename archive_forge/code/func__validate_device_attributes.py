from __future__ import (absolute_import, division, print_function)
import json
from ssl import SSLError
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME, ome_auth_params
from ansible.module_utils.urls import ConnectionError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def _validate_device_attributes(module):
    device_id_tags = []
    service_tag = module.params.get('device_service_tag')
    device_id = module.params.get('device_id')
    devices = module.params.get('devices')
    if devices:
        for dev in devices:
            if dev.get('id'):
                device_id_tags.append(dev.get('id'))
            else:
                device_id_tags.append(dev.get('service_tag'))
    if device_id is not None:
        device_id_tags.extend(device_id)
    if service_tag is not None:
        device_id_tags.extend(service_tag)
    return device_id_tags