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
def _find_error(self, response):
    """Searches the buffered response for a matching error condition"""
    for stderr_regex in self._terminal_stderr_re:
        if stderr_regex.search(response):
            self._log_messages("matched error regex (terminal_stderr_re) '%s' from response '%s'" % (stderr_regex.pattern, response))
            self._log_messages("matched stdout regex (terminal_stdout_re) '%s' from error response '%s'" % (self._matched_pattern, response))
            return True
    return False