from __future__ import absolute_import, division, print_function
import traceback
import re
import json
from itertools import chain
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils._text import to_native
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, _load_params
from ansible.module_utils.urls import open_url
def _connect_netbox_api(self, url, token, ssl_verify, cert):
    try:
        session = requests.Session()
        session.verify = ssl_verify
        if cert:
            session.cert = tuple((i for i in cert))
        nb = pynetbox.api(url, token=token)
        nb.http_session = session
        try:
            self.version = nb.version
            try:
                self.full_version = nb.status().get('netbox-version')
            except Exception:
                self.full_version = f'{self.version}.0'
        except AttributeError:
            self.module.fail_json(msg='Must have pynetbox >=4.1.0')
        except Exception:
            self.module.fail_json(msg='Failed to establish connection to NetBox API')
        return nb
    except Exception:
        self.module.fail_json(msg='Failed to establish connection to NetBox API')