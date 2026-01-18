from __future__ import absolute_import, division, print_function
import getpass
import json
import logging
import os
import re
import signal
import socket
import time
import traceback
from functools import wraps
from io import BytesIO
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import cPickle
from ansible.playbook.play_context import PlayContext
from ansible.plugins.loader import cache_loader, cliconf_loader, connection_loader, terminal_loader
from ansible_collections.ansible.netcommon.plugins.connection.libssh import HAS_PYLIBSSH
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.ansible.netcommon.plugins.plugin_utils.connection_base import (
def _get_log_channel(self):
    user = ''
    try:
        user = getpass.getuser()
    except KeyError:
        self.queue_message('vv', 'Current user (uid=%s) does not seem to exist on this system, leaving user empty.' % os.getuid())
    name = 'p=%s u=%s | ' % (os.getpid(), user)
    name += '%s [%s]' % (self.ssh_type, self._play_context.remote_addr)
    return name