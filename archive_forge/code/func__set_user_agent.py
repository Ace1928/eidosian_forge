from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _set_user_agent(clc):
    if hasattr(clc, 'SetRequestsSession'):
        agent_string = 'ClcAnsibleModule/' + __version__
        ses = requests.Session()
        ses.headers.update({'Api-Client': agent_string})
        ses.headers['User-Agent'] += ' ' + agent_string
        clc.SetRequestsSession(ses)