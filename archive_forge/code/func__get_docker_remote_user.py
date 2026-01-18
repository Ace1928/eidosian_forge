from __future__ import (absolute_import, division, print_function)
import fcntl
import os
import os.path
import subprocess
import re
from ansible.compat import selectors
from ansible.errors import AnsibleError, AnsibleFileNotFound
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase, BUFSIZE
from ansible.utils.display import Display
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
def _get_docker_remote_user(self):
    """ Get the default user configured in the docker container """
    container = self.get_option('remote_addr')
    if container in self._container_user_cache:
        return self._container_user_cache[container]
    p = subprocess.Popen([self.docker_cmd, 'inspect', '--format', '{{.Config.User}}', container], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    out = to_text(out, errors='surrogate_or_strict')
    if p.returncode != 0:
        display.warning(u'unable to retrieve default user from docker container: %s %s' % (out, to_text(err)))
        self._container_user_cache[container] = None
        return None
    user = out.strip() or u'root'
    self._container_user_cache[container] = user
    return user