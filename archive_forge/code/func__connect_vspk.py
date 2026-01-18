from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def _connect_vspk(self):
    """
        Connects to a Nuage API endpoint
        """
    try:
        if self.api_certificate and self.api_key:
            self.nuage_connection = VSPK.NUVSDSession(username=self.api_username, enterprise=self.api_enterprise, api_url=self.api_url, certificate=(self.api_certificate, self.api_key))
        else:
            self.nuage_connection = VSPK.NUVSDSession(username=self.api_username, password=self.api_password, enterprise=self.api_enterprise, api_url=self.api_url)
        self.nuage_connection.start()
    except BambouHTTPError as error:
        self.module.fail_json(msg='Unable to connect to the API URL with given username, password and enterprise: {0}'.format(error))