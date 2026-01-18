from __future__ import (absolute_import, division, print_function)
import os.path
from ansible import constants as C
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import MutableSequence
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
from ansible.plugins.loader import connection_loader
def _format_rsync_rsh_target(self, host, path, user):
    """ formats rsync rsh target, escaping ipv6 addresses if needed """
    user_prefix = ''
    if path.startswith('rsync://'):
        return path
    if self._remote_transport not in DOCKER + PODMAN + BUILDAH and user:
        user_prefix = '%s@' % (user,)
    if self._host_is_ipv6_address(host):
        return '[%s%s]:%s' % (user_prefix, host, path)
    return '%s%s:%s' % (user_prefix, host, path)