from __future__ import absolute_import, division, print_function
import datetime
import re
import time
import tarfile
from ansible.module_utils.urls import fetch_file
from ansible_collections.community.general.plugins.module_utils.redfish_utils import RedfishUtils
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse
def _find_updateservice_additional_uris(self):
    """Find & set WDC-specific update service URIs"""
    response = self.get_request(self.root_uri + self._update_uri())
    if response['ret'] is False:
        return response
    data = response['data']
    if 'Actions' not in data:
        return {'ret': False, 'msg': 'Service does not support SimpleUpdate'}
    if '#UpdateService.SimpleUpdate' not in data['Actions']:
        return {'ret': False, 'msg': 'Service does not support SimpleUpdate'}
    action = data['Actions']['#UpdateService.SimpleUpdate']
    if 'target' not in action:
        return {'ret': False, 'msg': 'Service does not support SimpleUpdate'}
    self.simple_update_uri = action['target']
    self.simple_update_status_uri = '{0}/Status'.format(self.simple_update_uri)
    if 'Oem' not in data['Actions']:
        return {'ret': False, 'msg': 'Service does not support OEM operations'}
    if 'WDC' not in data['Actions']['Oem']:
        return {'ret': False, 'msg': 'Service does not support WDC operations'}
    if '#UpdateService.FWActivate' not in data['Actions']['Oem']['WDC']:
        return {'ret': False, 'msg': 'Service does not support FWActivate'}
    action = data['Actions']['Oem']['WDC']['#UpdateService.FWActivate']
    if 'target' not in action:
        return {'ret': False, 'msg': 'Service does not support FWActivate'}
    self.firmware_activate_uri = action['target']
    return {'ret': True}