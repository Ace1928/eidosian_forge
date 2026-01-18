from __future__ import (absolute_import, division, print_function)
import json
import os
import time
from ssl import SSLError
from ansible_collections.dellemc.openmanage.plugins.module_utils.redfish import Redfish, redfish_auth_params
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def _get_update_service_target(obj, module):
    """Returns all the URI which is required for firmware update dynamically."""
    action_resp = obj.invoke_request('GET', '{0}{1}'.format(obj.root_uri, UPDATE_SERVICE))
    action_attr = action_resp.json_data['Actions']
    protocol = module.params['transfer_protocol']
    update_uri = None
    push_uri = action_resp.json_data.get('HttpPushUri')
    inventory_uri = action_resp.json_data.get('FirmwareInventory').get('@odata.id')
    if '#UpdateService.SimpleUpdate' in action_attr:
        update_service = action_attr.get('#UpdateService.SimpleUpdate')
        proto = update_service.get('TransferProtocol@Redfish.AllowableValues')
        if isinstance(proto, list) and protocol in proto and ('target' in update_service):
            update_uri = update_service.get('target')
        else:
            module.fail_json(msg='Target firmware version does not support {0} protocol.'.format(protocol))
    if update_uri is None or push_uri is None or inventory_uri is None:
        module.fail_json(msg='Target firmware version does not support redfish firmware update.')
    return (str(inventory_uri), str(push_uri), str(update_uri))