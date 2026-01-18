from __future__ import absolute_import, division, print_function
import stat
import os
import traceback
from ansible.module_utils.common import respawn
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils import distro
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
def _respawn_dnf():
    if respawn.has_respawned():
        return
    system_interpreters = ('/usr/libexec/platform-python', '/usr/bin/python3', '/usr/bin/python2', '/usr/bin/python')
    interpreter = respawn.probe_interpreters_for_module(system_interpreters, 'dnf')
    if interpreter:
        respawn.respawn_module(interpreter)