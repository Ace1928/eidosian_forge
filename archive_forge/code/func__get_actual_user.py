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
def _get_actual_user(self):
    if self.remote_user is not None:
        if self.docker_version == u'dev' or LooseVersion(self.docker_version) >= LooseVersion(u'1.7'):
            return self.remote_user
        else:
            self.remote_user = None
            actual_user = self._get_docker_remote_user()
            if actual_user != self.get_option('remote_user'):
                display.warning(u'docker {0} does not support remote_user, using container default: {1}'.format(self.docker_version, self.actual_user or u'?'))
            return actual_user
    elif self._display.verbosity > 2:
        return self._get_docker_remote_user()
    else:
        return None