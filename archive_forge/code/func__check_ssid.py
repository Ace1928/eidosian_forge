from __future__ import absolute_import, division, print_function
import json
import random
import mimetypes
from pprint import pformat
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils._text import to_native
def _check_ssid(self):
    """Verify storage system identifier exist on the proxy and, if not, then update to match storage system name."""
    try:
        rc, data = self._request(url=self.url + self.DEFAULT_REST_API_ABOUT_PATH, **self.creds)
        if data['runningAsProxy']:
            if self.ssid.lower() not in ['proxy', '0']:
                try:
                    rc, systems = self._request(url=self.url + self.DEFAULT_REST_API_PATH + 'storage-systems', **self.creds)
                    alternates = []
                    for system in systems:
                        if system['id'] == self.ssid:
                            break
                        elif system['name'] == self.ssid:
                            alternates.append(system['id'])
                    else:
                        if len(alternates) == 1:
                            self.module.warn('Array Id does not exist on Web Services Proxy Instance! However, there is a storage system with a matching name. Updating Identifier. Array Name: [%s], Array Id [%s].' % (self.ssid, alternates[0]))
                            self.ssid = alternates[0]
                        else:
                            self.module.fail_json(msg='Array identifier does not exist on Web Services Proxy Instance! Array ID [%s].' % self.ssid)
                except Exception as error:
                    self.module.fail_json(msg='Failed to determine Web Services Proxy storage systems! Array [%s]. Error [%s]' % (self.ssid, to_native(error)))
    except Exception as error:
        pass