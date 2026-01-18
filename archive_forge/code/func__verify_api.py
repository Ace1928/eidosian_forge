from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def _verify_api(self):
    """
        Verifies the API and loads the proper VSPK version
        """
    if ('api_password' not in list(self.auth.keys()) or not self.auth['api_password']) and ('api_certificate' not in list(self.auth.keys()) or 'api_key' not in list(self.auth.keys()) or (not self.auth['api_certificate']) or (not self.auth['api_key'])):
        self.module.fail_json(msg='Missing api_password or api_certificate and api_key parameter in auth')
    self.api_username = self.auth['api_username']
    if 'api_password' in list(self.auth.keys()) and self.auth['api_password']:
        self.api_password = self.auth['api_password']
    if 'api_certificate' in list(self.auth.keys()) and 'api_key' in list(self.auth.keys()) and self.auth['api_certificate'] and self.auth['api_key']:
        self.api_certificate = self.auth['api_certificate']
        self.api_key = self.auth['api_key']
    self.api_enterprise = self.auth['api_enterprise']
    self.api_url = self.auth['api_url']
    self.api_version = self.auth['api_version']
    try:
        global VSPK
        VSPK = importlib.import_module('vspk.{0:s}'.format(self.api_version))
    except ImportError:
        self.module.fail_json(msg='vspk is required for this module, or the API version specified does not exist.')