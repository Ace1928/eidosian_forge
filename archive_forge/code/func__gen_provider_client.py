from __future__ import (absolute_import, division, print_function)
import re
import time
import traceback
from ansible.module_utils.basic import (AnsibleModule, env_fallback,
from ansible.module_utils.common.text.converters import to_text
def _gen_provider_client(self):
    m = self._module
    p = {'auth_url': m.params['identity_endpoint'], 'password': m.params['password'], 'username': m.params['user'], 'project_name': m.params['project'], 'user_domain_name': m.params['domain'], 'reauthenticate': True}
    self._project_client = Adapter(session.Session(auth=v3.Password(**p)), raise_exc=False)
    p.pop('project_name')
    self._domain_client = Adapter(session.Session(auth=v3.Password(**p)), raise_exc=False)