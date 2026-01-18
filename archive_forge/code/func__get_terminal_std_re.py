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
def _get_terminal_std_re(self, option):
    terminal_std_option = self.get_option(option)
    terminal_std_re = []
    if terminal_std_option:
        for item in terminal_std_option:
            if 'pattern' not in item:
                raise AnsibleConnectionFailure("'pattern' is a required key for option '%s', received option value is %s" % (option, item))
            pattern = b'%s' % to_bytes(item['pattern'])
            flag = item.get('flags', 0)
            if flag:
                flag = getattr(re, flag.split('.')[1])
            terminal_std_re.append(re.compile(pattern, flag))
    else:
        terminal_std_re = getattr(self._terminal, option)
    return terminal_std_re